import numpy as np
import warnings
from numpy.linalg import norm, solve
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
import os
import sys
import sysconfig
import xml.etree.ElementTree as ET
import importlib.metadata
import importlib.util
from std_msgs.msg import Bool, String
try:
    import mujoco
except Exception:
    mujoco = None

PINK_IMPORT_ERROR = None


def prefer_current_python_site_packages():
    """Keep conda/venv packages ahead of system dist-packages in ROS launches."""
    conda_prefix = os.environ.get('CONDA_PREFIX')
    conda_site = None
    cmeel_python_sites = [os.path.join(
        os.path.expanduser('~'),
        '.local',
        'lib',
        f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages',
        'cmeel.prefix',
        'lib',
        f'python{sys.version_info.major}.{sys.version_info.minor}',
        'site-packages',
    )]
    if conda_prefix:
        conda_site = os.path.join(
            conda_prefix,
            'lib',
            f'python{sys.version_info.major}.{sys.version_info.minor}',
            'site-packages',
        )
        cmeel_python_sites.append(os.path.join(
            conda_site,
            'cmeel.prefix',
            'lib',
            f'python{sys.version_info.major}.{sys.version_info.minor}',
            'site-packages',
        ))
    numpy_site = os.path.dirname(os.path.dirname(np.__file__))
    purelib = sysconfig.get_paths().get('purelib')
    platlib = sysconfig.get_paths().get('platlib')
    preferred_paths = cmeel_python_sites + [conda_site, numpy_site, purelib, platlib]
    for path in reversed(preferred_paths):
        if path and os.path.isdir(path):
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)


prefer_current_python_site_packages()

def major_minor(version_text):
    parts = []
    for part in version_text.split('.')[:2]:
        number = ''.join(char for char in part if char.isdigit())
        parts.append(int(number) if number else 0)
    while len(parts) < 2:
        parts.append(0)
    return tuple(parts)


def scipy_numpy_abi_risky():
    try:
        numpy_version = major_minor(np.__version__)
        scipy_version = major_minor(importlib.metadata.version('scipy'))
        scipy_spec = importlib.util.find_spec('scipy')
    except Exception:
        return False

    scipy_origin = scipy_spec.origin if scipy_spec is not None else ''
    system_scipy = scipy_origin and '/usr/lib/python3/dist-packages' in scipy_origin
    return numpy_version >= (2, 0) and (scipy_version < (1, 13) or system_scipy)


if scipy_numpy_abi_risky():
    least_squares = None
else:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A NumPy version .* is required for this version of SciPy.*",
                category=UserWarning,
            )
            from scipy.optimize import least_squares
    except Exception:
        least_squares = None


from ament_index_python.packages import get_package_share_directory
arm_asset_pkg_path = get_package_share_directory('arm_asset')
# 模型路径
urdf_path = os.path.join(arm_asset_pkg_path, 'urdf', 'arm.urdf')

JOINT_NAMES = ['j1_joint', 'j2_joint', 'j3_joint', 'j4_joint', 'j5_joint', 'j6_joint']
JOINT_LIMIT_MARGIN = 1e-4
BASE_LINK = 'base_link'
EE_LINK = 'ee_center_link'
DOF = len(JOINT_NAMES)
ELBOW_BRANCH_MARGIN = 0.05
NEUTRAL_Q = np.array([0.0, -0.6, 0.8, 0.0, -0.5, 0.0], dtype=np.float64)
IK_ORIENTATION_TOL = np.deg2rad(10.0)
IK_POSITION_TOL = 0.01
IK_POLISH_CANDIDATES = 10
IK_LIMIT_AVOIDANCE_ZONE = 0.08
SELF_COLLISION_PENETRATION_TOL = 1e-4
J4_MAX_DELTA = 0.08
PINK_YAW_SAMPLES = 8
PINK_MAX_STARTS = 10
PINK_DT = 0.02
PINK_MAX_ITER = 120
PINK_POSITION_COST = 50.0
PINK_ORIENTATION_COST = 5.0
PINK_LM_DAMPING = 1e-4
PINK_POSTURE_COST = 0.05
COMBINED_POSITION_WEIGHT = 1.0
COMBINED_ORIENTATION_WEIGHT = 0.5
SOLUTION_J4_ABSOLUTE_WEIGHT = 0.5
IK_RESCUE_POSTURE_WEIGHT = 0.02
IK_RESCUE_NEUTRAL_WEIGHT = 0.01
IK_RESCUE_J4_WEIGHT = 0.05
IK_RESCUE_LIMIT_WEIGHT = 0.03

# J4优选范围（接近零位）
J4_PREFERRED_RANGE = (-0.1, 0.1)
J4_PENALTY_WEIGHT = 0.2  # J4运动惩罚权重

# 目标姿态：末端执行器竖直向下（Z轴指向地）
TARGET_ORIENTATION_WORLD = np.array([0, 0, 1])
OWNER_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


def parse_xyz(text, default):
    if text is None:
        return np.array(default, dtype=np.float64)
    return np.array([float(value) for value in text.split()], dtype=np.float64)


def rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(norm(axis), 1e-12)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ], dtype=np.float64)


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    return rotation_matrix([0.0, 0.0, 1.0], yaw) @ \
        rotation_matrix([0.0, 1.0, 0.0], pitch) @ \
        rotation_matrix([1.0, 0.0, 0.0], roll)


def homogeneous_transform(xyz, rpy):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def parse_urdf_kinematic_chain(path, base_link=BASE_LINK, ee_link=EE_LINK):
    """Parse the serial kinematic chain needed by this IK node from URDF."""
    root = ET.parse(path).getroot()
    joints_by_parent = {}

    for joint_el in root.findall('joint'):
        parent_el = joint_el.find('parent')
        child_el = joint_el.find('child')
        if parent_el is None or child_el is None:
            continue

        origin_el = joint_el.find('origin')
        origin_xyz = parse_xyz(origin_el.get('xyz') if origin_el is not None else None, [0.0, 0.0, 0.0])
        origin_rpy = parse_xyz(origin_el.get('rpy') if origin_el is not None else None, [0.0, 0.0, 0.0])
        axis_el = joint_el.find('axis')
        axis = parse_xyz(axis_el.get('xyz') if axis_el is not None else None, [0.0, 0.0, 1.0])

        limit_el = joint_el.find('limit')
        lower = float(limit_el.get('lower', '0.0')) if limit_el is not None else 0.0
        upper = float(limit_el.get('upper', '0.0')) if limit_el is not None else 0.0

        joint = {
            'name': joint_el.get('name'),
            'type': joint_el.get('type', 'fixed'),
            'parent': parent_el.get('link'),
            'child': child_el.get('link'),
            'origin': homogeneous_transform(origin_xyz, origin_rpy),
            'axis': axis,
            'lower': lower,
            'upper': upper,
        }
        joints_by_parent.setdefault(joint['parent'], []).append(joint)

    chain = []
    current_link = base_link
    visited = set()
    while current_link != ee_link:
        if current_link in visited:
            raise RuntimeError(f'URDF运动链存在环: {current_link}')
        visited.add(current_link)

        children = joints_by_parent.get(current_link, [])
        next_joint = None
        for joint in children:
            if joint['name'] in JOINT_NAMES or joint['type'] == 'fixed':
                next_joint = joint
                break

        if next_joint is None:
            raise RuntimeError(f'无法从 {current_link} 找到到 {ee_link} 的运动链')

        chain.append(next_joint)
        current_link = next_joint['child']

    movable_joints = [joint for joint in chain if joint['type'] != 'fixed']
    movable_names = [joint['name'] for joint in movable_joints]
    if movable_names != JOINT_NAMES:
        raise RuntimeError(f'URDF关节顺序不匹配: {movable_names}')

    lower_limits = np.array([joint['lower'] for joint in movable_joints], dtype=np.float64)
    upper_limits = np.array([joint['upper'] for joint in movable_joints], dtype=np.float64)
    return chain, lower_limits, upper_limits


KINEMATIC_CHAIN, JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS = parse_urdf_kinematic_chain(urdf_path)
JOINT_LOWER_LIMITS = JOINT_LOWER_LIMITS + JOINT_LIMIT_MARGIN
JOINT_UPPER_LIMITS = JOINT_UPPER_LIMITS - JOINT_LIMIT_MARGIN
mjcf_path = os.path.join(arm_asset_pkg_path, 'mjcf', 'arm_mjcf.xml')


def create_mujoco_collision_checker(path):
    if mujoco is None:
        return None, None
    try:
        model = mujoco.MjModel.from_xml_path(path)
        return model, mujoco.MjData(model)
    except Exception:
        return None, None


COLLISION_MODEL, COLLISION_DATA = create_mujoco_collision_checker(mjcf_path)


def create_pink_solver(path):
    """Create Pink/Pinocchio model and choose an installed QP backend."""
    global PINK_IMPORT_ERROR

    try:
        import pinocchio as pin
        from pink import Configuration, solve_ik as pink_solve_ik
        from pink.limits import ConfigurationLimit
        from pink.tasks import FrameTask, PostureTask
        import quadprog
    except Exception as exc:
        PINK_IMPORT_ERROR = str(exc)
        return None

    try:
        model = pin.buildModelFromUrdf(path)
        data = model.createData()
        ee_frame_id = model.getFrameId(EE_LINK)
        limit = ConfigurationLimit(model)
    except Exception as exc:
        PINK_IMPORT_ERROR = str(exc)
        return None

    qp_solver = 'quadprog'

    return {
        'pin': pin,
        'Configuration': Configuration,
        'solve_ik': pink_solve_ik,
        'ConfigurationLimit': ConfigurationLimit,
        'FrameTask': FrameTask,
        'PostureTask': PostureTask,
        'model': model,
        'data': data,
        'ee_frame_id': ee_frame_id,
        'limit': limit,
        'qp_solver': qp_solver,
    }


PINK_SOLVER = create_pink_solver(urdf_path)


def enforce_joint_limits(q):
    """Clamp joint angles to the limits declared in the URDF."""
    return np.clip(np.asarray(q, dtype=np.float64), JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS)


def check_self_collision(q):
    """Check joint limits and MuJoCo self-collision contacts."""
    q = np.asarray(q, dtype=np.float64)
    for i, (lower, upper) in enumerate(zip(JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS)):
        if q[i] < lower or q[i] > upper:
            return True, f"关节{i+1}超出限位"

    if COLLISION_MODEL is not None and COLLISION_DATA is not None:
        COLLISION_DATA.qpos[:DOF] = q
        COLLISION_DATA.qvel[:DOF] = 0.0
        mujoco.mj_forward(COLLISION_MODEL, COLLISION_DATA)

        for i in range(COLLISION_DATA.ncon):
            contact = COLLISION_DATA.contact[i]
            if contact.dist >= -SELF_COLLISION_PENETRATION_TOL:
                continue

            body1 = mujoco.mj_id2name(
                COLLISION_MODEL,
                mujoco.mjtObj.mjOBJ_BODY,
                COLLISION_MODEL.geom_bodyid[contact.geom1],
            )
            body2 = mujoco.mj_id2name(
                COLLISION_MODEL,
                mujoco.mjtObj.mjOBJ_BODY,
                COLLISION_MODEL.geom_bodyid[contact.geom2],
            )
            return True, f"自碰撞: {body1} 与 {body2}, 穿透={-contact.dist:.4f}m"

    return False, "无碰撞"


def forward_kinematics(q):
    """Return end-effector transform plus revolute joint origins/axes in world frame."""
    q = np.asarray(q, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    joint_origins = []
    joint_axes = []
    joint_index = 0

    for joint in KINEMATIC_CHAIN:
        transform = transform @ joint['origin']

        if joint['type'] == 'fixed':
            continue

        axis_world = transform[:3, :3] @ joint['axis']
        axis_world = axis_world / max(norm(axis_world), 1e-12)
        joint_origins.append(transform[:3, 3].copy())
        joint_axes.append(axis_world.copy())

        joint_rotation = np.eye(4, dtype=np.float64)
        joint_rotation[:3, :3] = rotation_matrix(joint['axis'], q[joint_index])
        transform = transform @ joint_rotation
        joint_index += 1

    return transform, joint_origins, joint_axes


def fk_pose(q, solver_data=None):
    """Return end-effector pose (position and orientation) in world coordinates."""
    if PINK_SOLVER is not None:
        pin = PINK_SOLVER['pin']
        model = PINK_SOLVER['model']
        solver_data = solver_data if solver_data is not None else model.createData()
        pin.framesForwardKinematics(model, solver_data, np.asarray(q, dtype=np.float64))
        transform = solver_data.oMf[PINK_SOLVER['ee_frame_id']].homogeneous
    else:
        transform, _, _ = forward_kinematics(q)
    position = transform[:3, 3].copy()
    # 获取末端执行器的Z轴方向（在全局坐标系中）
    z_axis = transform[:3, :3][:, 2].copy()  # 第三列是局部Z轴在全局中的表示
    return position, z_axis


def compute_ee_jacobian(q):
    """Geometric Jacobian for the end-effector in world frame."""
    if PINK_SOLVER is not None:
        pin = PINK_SOLVER['pin']
        model = PINK_SOLVER['model']
        solver_data = model.createData()
        q = np.asarray(q, dtype=np.float64)
        pin.computeJointJacobians(model, solver_data, q)
        pin.updateFramePlacements(model, solver_data)
        return pin.getFrameJacobian(
            model,
            solver_data,
            PINK_SOLVER['ee_frame_id'],
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

    transform, joint_origins, joint_axes = forward_kinematics(q)
    ee_position = transform[:3, 3]
    jacobian = np.zeros((6, DOF), dtype=np.float64)

    for i, (origin, axis) in enumerate(zip(joint_origins, joint_axes)):
        jacobian[:3, i] = np.cross(axis, ee_position - origin)
        jacobian[3:6, i] = axis

    return jacobian


def fk_position(q, solver_data=None):
    """Return end-effector position only (for backward compatibility)."""
    pos, _ = fk_pose(q, solver_data)
    return pos


def position_error(q, target_position, solver_data=None):
    """Return the true Cartesian position error in meters."""
    return norm(np.asarray(target_position, dtype=np.float64) - fk_position(q, solver_data))


def orientation_error(q, target_z_axis=TARGET_ORIENTATION_WORLD, solver_data=None):
    """Calculate orientation error: angle between current_z and target_z."""
    _, current_z = fk_pose(q, solver_data)
    dot_product = np.clip(np.dot(current_z, target_z_axis), -1, 1)
    # 使用角度误差（弧度）更准确
    return np.arccos(dot_product)


def axis_alignment_error(current_z, target_z=TARGET_ORIENTATION_WORLD):
    """Return angular-velocity direction that aligns current_z with target_z."""
    current_z = np.asarray(current_z, dtype=np.float64)
    target_z = np.asarray(target_z, dtype=np.float64)
    current_z = current_z / max(norm(current_z), 1e-12)
    target_z = target_z / max(norm(target_z), 1e-12)
    error = np.cross(current_z, target_z)

    # The cross product vanishes at 180 degrees even though the pose is wrong.
    if norm(error) < 1e-8 and np.dot(current_z, target_z) < 0.0:
        basis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(current_z, basis)) > 0.9:
            basis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        error = np.cross(current_z, basis)
        error = error / max(norm(error), 1e-12)

    return error


def joint_limit_avoidance_residual(q):
    """Smoothly penalize configurations that sit close to joint limits."""
    q = np.asarray(q, dtype=np.float64)
    span = JOINT_UPPER_LIMITS - JOINT_LOWER_LIMITS
    lower_distance = (q - JOINT_LOWER_LIMITS) / span
    upper_distance = (JOINT_UPPER_LIMITS - q) / span
    lower_penalty = np.maximum(0.0, IK_LIMIT_AVOIDANCE_ZONE - lower_distance)
    upper_penalty = np.maximum(0.0, IK_LIMIT_AVOIDANCE_ZONE - upper_distance)
    return np.concatenate([lower_penalty, upper_penalty]) / IK_LIMIT_AVOIDANCE_ZONE


def combined_error(
        q,
        target_position,
        pos_weight=COMBINED_POSITION_WEIGHT,
        ori_weight=COMBINED_ORIENTATION_WEIGHT,
        solver_data=None):
    """Combined position and orientation error."""
    pos_error = norm(target_position - fk_position(q, solver_data))
    ori_error = orientation_error(q, TARGET_ORIENTATION_WORLD, solver_data)
    return pos_weight * pos_error + ori_weight * ori_error


def solution_sort_key(q, target_position, reference_q, eps, solver_data=None):
    """
    Prefer accurate IK solutions, then smooth branch changes.
    J4零位偏好只在位置/姿态满足后参与排序，避免牺牲可达性。
    """
    pos_err = position_error(q, target_position, solver_data)
    pos_key = 0.0 if pos_err < IK_POSITION_TOL else pos_err
    ori_err = orientation_error(q, TARGET_ORIENTATION_WORLD, solver_data)
    ori_key = 0.0 if ori_err <= IK_ORIENTATION_TOL else ori_err
    feasible_key = 0 if pos_key == 0.0 and ori_key == 0.0 else 1
    normalized_violation = max(
        pos_err / IK_POSITION_TOL,
        ori_err / IK_ORIENTATION_TOL,
    )
    motion = norm(q - reference_q) if reference_q is not None else 0.0
    j4_delta = abs(q[3] - reference_q[3]) if reference_q is not None else abs(q[3])
    j4_large_move = max(0.0, j4_delta - J4_MAX_DELTA)
    j4_penalty = abs(q[3]) * SOLUTION_J4_ABSOLUTE_WEIGHT
    if reference_q is not None:
        j4_penalty += j4_delta * J4_PENALTY_WEIGHT
    elbow_penalty = max(0.0, -q[2] - ELBOW_BRANCH_MARGIN)
    return (
        feasible_key,
        normalized_violation if feasible_key else 0.0,
        j4_large_move,
        j4_delta,
        pos_err,
        ori_err,
        j4_penalty,
        motion,
        elbow_penalty,
    )


def is_collision_free_solution(q, reference_q=None):
    """Check the candidate and the direct joint-space path used by the controller."""
    q = np.asarray(q, dtype=np.float64)
    if reference_q is None:
        samples = [q]
    else:
        reference_q = np.asarray(reference_q, dtype=np.float64)
        sample_count = max(2, int(np.ceil(np.max(np.abs(q - reference_q)) / 0.05)) + 1)
        samples = (
            reference_q + alpha * (q - reference_q)
            for alpha in np.linspace(0.0, 1.0, sample_count)
        )

    return all(not check_self_collision(sample)[0] for sample in samples)


def make_seed_configs(target_position, current_q=None):
    """
    Generate deterministic seeds covering elbow and base-yaw branches.
    减少J4的多样性，优先使用零位
    """
    seeds = []
    seen = set()

    def add(q):
        q = enforce_joint_limits(q)
        key = tuple(np.round(q, 5))
        if key not in seen:
            seen.add(key)
            seeds.append(q)

    if current_q is not None:
        add(current_q)

    # 中性姿态（J4保持零位）
    add(NEUTRAL_Q)

    # 如果当前J4不为零，添加一个J4为零的变体
    if current_q is not None and abs(current_q[3]) > 0.05:
        add(NEUTRAL_Q)

    x, y, _ = target_position
    yaw_guess = np.arctan2(y - 0.08, x - 0.08)
    yaw_guesses = [
        0.0,
        yaw_guess,
        yaw_guess + np.pi / 2.0,
        yaw_guess - np.pi / 2.0,
        yaw_guess + np.pi,
        -yaw_guess,
    ]

    j2_guesses = [-0.3, -0.6, -0.9, -1.2, -1.6, -2.0]
    j3_guesses = [0.3, 0.6, 0.9, 1.2, 1.5]

    # J4只使用零位，减少多样性
    j4_guesses = [0.0]
    # 如果当前J4不为零，才添加当前值作为备选
    if current_q is not None and abs(current_q[3]) > 0.1:
        j4_guesses.append(current_q[3])

    # J5/J6组合减少
    wrist_guesses = [(-0.5, 0.0), (0.0, 0.0), (0.5, 0.0)]

    for j1 in yaw_guesses:
        for j2 in j2_guesses:
            for j3 in j3_guesses:
                # 保持J4零位
                add(np.array([j1, j2, j3, 0.0, -0.5, 0.0], dtype=np.float64))

    for j1 in yaw_guesses[:4]:
        for j2 in [-0.5, -0.9, -1.3, -1.7]:
            for j3 in [0.4, 0.8, 1.2]:
                for j4 in j4_guesses:
                    for j5, j6 in wrist_guesses:
                        add(np.array([j1, j2, j3, j4, j5, j6], dtype=np.float64))

    return seeds


def rotation_with_z_axis(target_z, yaw):
    """Build a rotation matrix whose local Z axis equals target_z."""
    z_axis = np.asarray(target_z, dtype=np.float64)
    z_axis = z_axis / max(norm(z_axis), 1e-12)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(helper, z_axis)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    x_axis = helper - np.dot(helper, z_axis) * z_axis
    x_axis = x_axis / max(norm(x_axis), 1e-12)
    y_axis = np.cross(z_axis, x_axis)

    yaw_rotation = rotation_matrix(z_axis, yaw)
    x_axis = yaw_rotation @ x_axis
    y_axis = yaw_rotation @ y_axis
    return np.column_stack([x_axis, y_axis, z_axis])


def select_pink_seeds(seeds):
    """Keep Pink solve time bounded while preserving branch diversity."""
    if len(seeds) <= PINK_MAX_STARTS:
        return seeds

    selected = seeds[:2]
    remaining_slots = PINK_MAX_STARTS - len(selected)
    stride = max(1, (len(seeds) - 2) // remaining_slots)
    selected.extend(seeds[2::stride][:remaining_slots])
    return selected


def make_pink_target_poses(target_position):
    """Pink solves full-pose IK, so sample the free yaw around target Z."""
    poses = []
    pin = PINK_SOLVER['pin']
    for yaw in np.linspace(-np.pi, np.pi, PINK_YAW_SAMPLES, endpoint=False):
        rotation = rotation_with_z_axis(TARGET_ORIENTATION_WORLD, yaw)
        poses.append(pin.SE3(rotation, np.asarray(target_position, dtype=np.float64)))
    return poses


def solve_ik_pink(target_position, seeds, eps, max_iter, reference_q=None):
    """Pink task-space IK solver with multi-start and free-yaw sampling."""
    best_q = None
    best_error = float('inf')
    best_key = None

    if reference_q is None:
        reference_q = NEUTRAL_Q
    reference_q = enforce_joint_limits(reference_q)

    model = PINK_SOLVER['model']
    Configuration = PINK_SOLVER['Configuration']
    FrameTask = PINK_SOLVER['FrameTask']
    PostureTask = PINK_SOLVER['PostureTask']
    pink_solve_ik = PINK_SOLVER['solve_ik']
    qp_solver = PINK_SOLVER['qp_solver']
    limits = [PINK_SOLVER['limit']]
    iteration_count = min(max_iter, PINK_MAX_ITER)

    for seed in select_pink_seeds(seeds):
        for pose in make_pink_target_poses(target_position):
            configuration = Configuration(model, model.createData(), enforce_joint_limits(seed))
            frame_task = FrameTask(
                EE_LINK,
                position_cost=PINK_POSITION_COST,
                orientation_cost=PINK_ORIENTATION_COST,
                lm_damping=PINK_LM_DAMPING,
            )
            frame_task.set_target(pose)
            posture_task = PostureTask(cost=PINK_POSTURE_COST)
            posture_task.set_target(reference_q)
            tasks = [frame_task, posture_task]

            for _ in range(iteration_count):
                try:
                    velocity = pink_solve_ik(
                        configuration,
                        tasks,
                        dt=PINK_DT,
                        solver=qp_solver,
                        damping=1e-6,
                        limits=limits,
                    )
                    configuration.integrate_inplace(velocity, PINK_DT)
                    configuration.update(enforce_joint_limits(configuration.q))
                except Exception:
                    break

                pos_err = position_error(configuration.q, target_position)
                ori_err = orientation_error(configuration.q, TARGET_ORIENTATION_WORLD)
                if pos_err < eps and ori_err < IK_ORIENTATION_TOL:
                    break

            q = enforce_joint_limits(configuration.q)
            pos_err = position_error(q, target_position)
            if not is_collision_free_solution(q, reference_q):
                continue
            key = solution_sort_key(q, target_position, reference_q, eps)

            if best_key is None or key < best_key:
                best_q = q.copy()
                best_error = pos_err
                best_key = key

    return best_q, best_error


def solve_ik_scipy(target_position, seeds, eps, max_iter, reference_q=None):
    """
    Hybrid IK solver:
    1. Run a fast damped-least-squares pass from every seed.
    2. Polish the best candidates with bounded trust-region least-squares.

    This keeps the multi-start robustness of the old solver while spending the
    slower SciPy optimization budget only on promising branches.
    """
    best_q = None
    best_error = float('inf')
    best_key = None

    if reference_q is None:
        reference_q = NEUTRAL_Q
    reference_q = enforce_joint_limits(reference_q)

    candidates = []
    dls_iter = min(max_iter, max(60, max_iter // 3))

    for seed in seeds:
        q, pos_err, key = solve_ik_dls_single(
            target_position,
            seed,
            eps,
            dls_iter,
            reference_q,
        )
        if q is None:
            continue
        if not is_collision_free_solution(q, reference_q):
            continue
        candidates.append((key, q, pos_err))
        if best_key is None or key < best_key:
            best_q = q.copy()
            best_error = pos_err
            best_key = key

    if not candidates:
        return best_q, best_error

    candidates.sort(key=lambda item: item[0])
    polish_starts = []
    seen = set()
    for _, q, _ in candidates[:IK_POLISH_CANDIDATES]:
        key = tuple(np.round(q, 5))
        if key not in seen:
            seen.add(key)
            polish_starts.append(q)

    pos_scale = max(eps, 1e-3)
    ori_scale = max(np.sin(IK_ORIENTATION_TOL), 1e-3)
    for seed in polish_starts:
        solver_data = None

        def residual(q):
            current_pos, current_z = fk_pose(q, solver_data)
            pos_residual = (current_pos - target_position) / pos_scale
            ori_residual = axis_alignment_error(current_z) / ori_scale
            posture_residual = IK_RESCUE_POSTURE_WEIGHT * (q - reference_q)
            neutral_residual = IK_RESCUE_NEUTRAL_WEIGHT * (q - NEUTRAL_Q)
            j4_residual = np.array([IK_RESCUE_J4_WEIGHT * q[3]])
            limit_residual = (
                IK_RESCUE_LIMIT_WEIGHT * joint_limit_avoidance_residual(q)
            )

            return np.concatenate([
                pos_residual,
                ori_residual,
                posture_residual,
                neutral_residual,
                j4_residual,
                limit_residual,
            ])

        try:
            result = least_squares(
                residual,
                seed,
                bounds=(JOINT_LOWER_LIMITS, JOINT_UPPER_LIMITS),
                method='trf',
                x_scale='jac',
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                max_nfev=max(50, max_iter // 2),
            )
        except ValueError:
            continue

        q = enforce_joint_limits(result.x)
        pos_err = position_error(q, target_position, solver_data)
        if not is_collision_free_solution(q, reference_q):
            continue
        key = solution_sort_key(q, target_position, reference_q, eps, solver_data)

        if best_key is None or key < best_key:
            best_q = q.copy()
            best_error = pos_err
            best_key = key

    return best_q, best_error


def solve_ik_orientation_rescue(target_position, seeds, reference_q, max_iter):
    """Find a publishable pose while strictly limiting J4 motion."""
    if least_squares is None:
        return None, float('inf')

    reference_q = enforce_joint_limits(reference_q)
    lower_limits = JOINT_LOWER_LIMITS.copy()
    upper_limits = JOINT_UPPER_LIMITS.copy()
    lower_limits[3] = max(lower_limits[3], reference_q[3] - J4_MAX_DELTA)
    upper_limits[3] = min(upper_limits[3], reference_q[3] + J4_MAX_DELTA)
    best_q = None
    best_error = float('inf')
    best_key = None

    for seed in seeds:
        seed = np.clip(seed, lower_limits, upper_limits)

        def residual(q):
            current_position, current_z = fk_pose(q)
            return np.concatenate([
                50.0 * (current_position - target_position),
                8.0 * axis_alignment_error(current_z),
                0.03 * (q - reference_q),
            ])

        try:
            result = least_squares(
                residual,
                seed,
                bounds=(lower_limits, upper_limits),
                method='trf',
                max_nfev=max(100, min(200, max_iter)),
            )
        except ValueError:
            continue

        q = enforce_joint_limits(result.x)
        pos_err = position_error(q, target_position)
        ori_err = orientation_error(q)
        if pos_err >= IK_POSITION_TOL or ori_err > IK_ORIENTATION_TOL:
            continue
        if not is_collision_free_solution(q, reference_q):
            continue

        key = (pos_err, ori_err, abs(q[3] - reference_q[3]), norm(q - reference_q))
        if best_key is None or key < best_key:
            best_q = q.copy()
            best_error = pos_err
            best_key = key

    return best_q, best_error


def solve_ik_dls_single(target_position, q_init, eps, max_iter, reference_q=None):
    """Damped least-squares IK from one seed, with null-space posture control."""
    if reference_q is None:
        reference_q = NEUTRAL_Q
    reference_q = enforce_joint_limits(reference_q)

    q = enforce_joint_limits(q_init)
    solver_data = None
    best_q = q.copy()
    best_error = float('inf')
    best_key = None
    damping = 1e-3
    pos_scale = max(eps, 1e-3)
    ori_scale = max(np.sin(IK_ORIENTATION_TOL), 1e-3)

    for _ in range(max_iter):
        current_pos, current_z = fk_pose(q, solver_data)
        pos_error = target_position - current_pos
        pos_error_norm = norm(pos_error)
        ori_error = axis_alignment_error(current_z)
        ori_error_norm = norm(ori_error)

        key = solution_sort_key(q, target_position, reference_q, eps, solver_data)
        if best_key is None or key < best_key:
            best_error = pos_error_norm
            best_q = q.copy()
            best_key = key

        if pos_error_norm < eps and ori_error_norm < np.sin(IK_ORIENTATION_TOL):
            break

        frame_jacobian = compute_ee_jacobian(q)

        task_error = np.concatenate([pos_error / pos_scale, ori_error / ori_scale])
        task_jacobian = np.vstack([
            frame_jacobian[:3, :] / pos_scale,
            frame_jacobian[3:6, :] / ori_scale,
        ])

        combined_err = norm(task_error)
        jj_t = task_jacobian @ task_jacobian.T
        adaptive_damping = damping * (1.0 + 0.2 * combined_err)
        jj_t += (adaptive_damping ** 2) * np.eye(jj_t.shape[0])

        try:
            dq_task = task_jacobian.T @ solve(jj_t, task_error)
            j_pinv = task_jacobian.T @ solve(jj_t, np.eye(task_jacobian.shape[0]))
            null_projector = np.eye(DOF) - j_pinv @ task_jacobian
        except np.linalg.LinAlgError:
            break

        dq_posture = 0.06 * (reference_q - q) + 0.02 * (NEUTRAL_Q - q)
        dq_posture[3] += 0.08 * (0.0 - q[3])
        dq = dq_task + null_projector @ dq_posture

        dq = np.clip(dq, -0.15, 0.15)
        q = enforce_joint_limits(q + dq)

    return best_q, best_error, best_key


def solve_ik_dls(target_position, seeds, eps, max_iter, reference_q=None):
    """Fallback multi-start damped least-squares solver."""
    best_q = None
    best_error = float('inf')
    best_key = None

    if reference_q is None:
        reference_q = NEUTRAL_Q
    reference_q = enforce_joint_limits(reference_q)

    for seed in seeds:
        q, pos_err, key = solve_ik_dls_single(
            target_position,
            seed,
            eps,
            max_iter,
            reference_q,
        )
        if q is not None and is_collision_free_solution(q, reference_q) and (best_key is None or key < best_key):
            best_q = q.copy()
            best_error = pos_err
            best_key = key

    return best_q, best_error


def multi_start_ik(target_position, current_q=None, max_iter=500, eps=0.008):
    """Multi-start IK with orientation constraint and J4 optimization."""
    target_position = np.asarray(target_position, dtype=np.float64)
    if current_q is not None:
        current_q = enforce_joint_limits(current_q)

    seeds = make_seed_configs(target_position, current_q)
    reference_q = current_q if current_q is not None else seeds[0]

    print(f"目标位置: ({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")
    if PINK_SOLVER is not None:
        solver_name = f"Pink ({PINK_SOLVER['qp_solver']})"
    elif PINK_IMPORT_ERROR:
        solver_name = f'Pink unavailable ({PINK_IMPORT_ERROR}), fallback'
    elif least_squares is not None:
        solver_name = 'NumPy DLS + scipy trust-region fallback'
    else:
        solver_name = 'NumPy DLS fallback'
    print(f"IK种子数量: {len(seeds)}, 求解器: {solver_name}")

    if PINK_SOLVER is not None:
        pink_q, pink_error = solve_ik_pink(target_position, seeds, eps, max_iter, reference_q)
        if (
            pink_q is not None
            and pink_error < IK_POSITION_TOL
            and orientation_error(pink_q) <= IK_ORIENTATION_TOL
        ):
            return pink_q, pink_error

        rescue_q, rescue_error = solve_ik_orientation_rescue(
            target_position,
            seeds,
            reference_q,
            max_iter,
        )
        if rescue_q is not None:
            return rescue_q, rescue_error

    if least_squares is not None:
        return solve_ik_scipy(target_position, seeds, eps, max_iter, reference_q)

    return solve_ik_dls(target_position, seeds, eps, max_iter, reference_q)


class ArmIKController(Node):
    def __init__(self):
        super().__init__('ik_control')

        # 订阅者
        self.goal_sub = self.create_subscription(Point, '/goal', self.goal_callback, 50)
        self.joint_sub = self.create_subscription(JointState, '/robot_joint_state', self.joint_callback, 10)
        self.mujoco_joint_sub = self.create_subscription(JointState, '/mujoco_joint_state', self.joint_callback, 10)

        # 发布者
        self.joint_pub = self.create_publisher(JointState, '/control', 10)
        self.handeye_joint_pub = self.create_publisher(JointState, '/handeye/control', 10)
        self.ik_success_pub = self.create_publisher(Bool, '/ik_success', 1)
        self.handeye_result_pub = self.create_publisher(String, '/handeye/ik_result', 10)
        self.handeye_goal_sub = self.create_subscription(
            String,
            '/handeye/goal_request',
            self.handeye_goal_callback,
            10,
        )
        self.handeye_active_sub = self.create_subscription(
            Bool,
            '/handeye/calibration_active',
            self.handeye_active_callback,
            OWNER_QOS,
        )
        # 状态变量
        self.goal = np.zeros(3, dtype=np.float64)
        self.current_joints = np.zeros(6, dtype=np.float64)
        self.has_joint_state = False
        self.handeye_active = False
        self.handeye_request_seq = None
        self.last_handeye_seq = -1

        # 统计
        self.stats = {'total': 0, 'success': 0, 'collision': 0, 'failure': 0}

        # 工作空间
        self.workspace = {
            'x': (-0.22, 0.38),
            'y': (-0.22, 0.38),
            'z': (0.05, 0.33)
        }

        # J4运动统计
        self.j4_movement_history = []
        self.last_j4 = None
        if PINK_SOLVER is not None:
            self.get_logger().info(f"IK solver: Pink ({PINK_SOLVER['qp_solver']})")
        elif PINK_IMPORT_ERROR:
            fallback_name = 'NumPy/SciPy fallback求解器' if least_squares is not None else 'NumPy DLS fallback求解器'
            self.get_logger().warn(f'Pink不可用: {PINK_IMPORT_ERROR}; 使用{fallback_name}')
        elif least_squares is not None:
            self.get_logger().warn('Pink未安装，使用NumPy/SciPy fallback求解器')
        else:
            self.get_logger().warn('Pink和SciPy均不可用，使用NumPy DLS fallback求解器')
        print("=" * 60)

    def joint_callback(self, msg):
        if len(msg.position) >= 6:
            if msg.name:
                name_aliases = {
                    'j1_joint': ('j1_joint', 'joint1'),
                    'j2_joint': ('j2_joint', 'joint2'),
                    'j3_joint': ('j3_joint', 'joint3'),
                    'j4_joint': ('j4_joint', 'joint4'),
                    'j5_joint': ('j5_joint', 'joint5'),
                    'j6_joint': ('j6_joint', 'joint6'),
                }
                name_to_position = dict(zip(msg.name, msg.position))
                joints = []
                for joint_name in JOINT_NAMES:
                    value = None
                    for alias in name_aliases[joint_name]:
                        if alias in name_to_position:
                            value = name_to_position[alias]
                            break
                    if value is None:
                        joints = None
                        break
                    joints.append(value)

                if joints is not None:
                    self.current_joints = enforce_joint_limits(np.array(joints, dtype=np.float64))
                    self.has_joint_state = True
                    return

            self.current_joints = enforce_joint_limits(np.array(msg.position[:6], dtype=np.float64))
            self.has_joint_state = True

    def check_workspace(self, pos):
        x_min, x_max = self.workspace['x']
        y_min, y_max = self.workspace['y']
        z_min, z_max = self.workspace['z']

        if not (x_min <= pos[0] <= x_max):
            return False
        if not (y_min <= pos[1] <= y_max):
            return False
        if not (z_min <= pos[2] <= z_max):
            return False
        return True

    def handeye_active_callback(self, msg):
        self.handeye_active = bool(msg.data)
        if not self.handeye_active:
            self.handeye_request_seq = None

    def publish_handeye_result(self, seq, success, reason):
        result = String()
        result.data = json.dumps({
            'seq': int(seq),
            'success': bool(success),
            'reason': str(reason),
        }, sort_keys=True)
        self.handeye_result_pub.publish(result)

    def publish_ik_result(self, success, reason=''):
        result = Bool()
        result.data = success
        self.ik_success_pub.publish(result)
        if self.handeye_request_seq is not None:
            self.publish_handeye_result(
                self.handeye_request_seq,
                success,
                reason or ('ok' if success else 'ik_failed'),
            )

    def handeye_goal_callback(self, msg):
        try:
            request = json.loads(msg.data)
            seq = request['seq']
            if not isinstance(seq, int) or isinstance(seq, bool) or seq <= self.last_handeye_seq:
                raise ValueError('seq must be a strictly increasing integer')
            goal = np.asarray(
                [request['x'], request['y'], request['z']],
                dtype=np.float64,
            )
            if goal.shape != (3,) or not np.all(np.isfinite(goal)):
                raise ValueError('goal must contain finite x/y/z')
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.get_logger().warn(f'忽略无效标定 IK 请求: {exc}')
            return
        if not self.handeye_active:
            self.publish_handeye_result(seq, False, 'calibration_owner_inactive')
            return
        self.last_handeye_seq = seq
        self.handeye_request_seq = seq
        self.goal[:] = goal
        self.stats['total'] += 1
        if not self.check_workspace(self.goal):
            self.publish_ik_result(False, 'target_out_of_workspace')
            self.handeye_request_seq = None
            return
        self.ik_control()
        self.handeye_request_seq = None

    def goal_callback(self, msg):
        if self.handeye_active:
            self.get_logger().warn('标定持有控制权，忽略普通 /goal')
            return
        self.handeye_request_seq = None
        self.goal[0] = msg.x
        self.goal[1] = msg.y
        self.goal[2] = msg.z

        self.stats['total'] += 1

        if not self.check_workspace(self.goal):
            self.get_logger().warn(f'目标#{self.stats["total"]}超出工作空间')
            self.publish_ik_result(False)
            return

        self.get_logger().info(f'\n📌 目标#{self.stats["total"]}: ({self.goal[0]:.3f}, {self.goal[1]:.3f}, {self.goal[2]:.3f})')
        self.ik_control()

    def goal_pose_callback(self, msg):
        """支持完整位姿的目标（可选）"""
        if self.handeye_active:
            self.get_logger().warn('标定持有控制权，忽略普通位姿目标')
            return
        self.handeye_request_seq = None
        self.goal[0] = msg.position.x
        self.goal[1] = msg.position.y
        self.goal[2] = msg.position.z

        self.stats['total'] += 1

        if not self.check_workspace(self.goal):
            self.get_logger().warn(f'目标#{self.stats["total"]}超出工作空间')
            self.publish_ik_result(False)
            return

        self.get_logger().info(f'\n📌 目标#{self.stats["total"]}: ({self.goal[0]:.3f}, {self.goal[1]:.3f}, {self.goal[2]:.3f})')
        self.ik_control()

    def ik_control(self):
        current_q = self.current_joints if self.has_joint_state else None
        ik_joint_pos, solver_error = multi_start_ik(self.goal, current_q=current_q, max_iter=300)
        pos_error = float('inf')
        orientation_angle = float('inf')
        orientation_deg = float('inf')

        if ik_joint_pos is not None:
            ik_joint_pos = enforce_joint_limits(ik_joint_pos)
            pos_error = position_error(ik_joint_pos, self.goal)
            # 验证最终姿态
            _, final_z = fk_pose(ik_joint_pos)
            dot_product = np.dot(final_z, TARGET_ORIENTATION_WORLD)
            orientation_angle = np.arccos(np.clip(dot_product, -1, 1))
            orientation_deg = np.degrees(orientation_angle)

            # 记录J4运动
            if self.has_joint_state:
                j4_movement = abs(ik_joint_pos[3] - self.current_joints[3])
                self.j4_movement_history.append(j4_movement)
                self.last_j4 = ik_joint_pos[3]

        if (
            ik_joint_pos is not None
            and pos_error < IK_POSITION_TOL
            and orientation_angle <= IK_ORIENTATION_TOL
        ):
            has_collision, reason = check_self_collision(ik_joint_pos)

            if has_collision:
                self.stats['collision'] += 1
                self.get_logger().warn(f'限位/碰撞检查失败: {reason}')
                self.publish_ik_result(False, 'collision')
                return

            # 发布控制命令
            cmd = JointState()
            cmd.header.stamp = self.get_clock().now().to_msg()
            cmd.name = JOINT_NAMES
            cmd.position = ik_joint_pos.tolist()
            if self.handeye_request_seq is None:
                self.joint_pub.publish(cmd)
            else:
                self.handeye_joint_pub.publish(cmd)

            self.stats['success'] += 1
            angles_deg = np.degrees(ik_joint_pos)

            # 输出J4信息
            j4_info = ""
            if self.has_joint_state:
                j4_movement = abs(ik_joint_pos[3] - self.current_joints[3])
                j4_info = f", J4运动={np.degrees(j4_movement):.1f}°"
            self.publish_ik_result(True)


            self.get_logger().info(
                f'✅ 成功! 位置误差={pos_error:.4f}m, 姿态误差={orientation_deg:.1f}°{j4_info}'
            )
            self.get_logger().info(f'   关节: j1={angles_deg[0]:.0f}°, j2={angles_deg[1]:.0f}°, '
                                   f'j3={angles_deg[2]:.0f}°, j4={angles_deg[3]:.0f}°, '
                                   f'j5={angles_deg[4]:.0f}°, j6={angles_deg[5]:.0f}°')
        else:
            self.stats['failure'] += 1
            self.publish_ik_result(False, 'final_tolerance_failed')
            self.get_logger().error(
                f'❌ 失败, 位置误差={pos_error:.4f}m, '
                f'姿态误差={orientation_deg:.1f}°, 求解代价={solver_error:.4f}m'
            )

    def __del__(self):
        if self.j4_movement_history:
            avg_j4_movement = np.mean(self.j4_movement_history)
            max_j4_movement = np.max(self.j4_movement_history)
            self.get_logger().info(
                f'\n📊 统计: 成功={self.stats["success"]}/{self.stats["total"]}, '
                f'平均J4运动={np.degrees(avg_j4_movement):.1f}°, '
                f'最大J4运动={np.degrees(max_j4_movement):.1f}°'
            )
        else:
            self.get_logger().info(f'\n📊 统计: 成功={self.stats["success"]}/{self.stats["total"]}')


def main(args=None):
    try:
        rclpy.init(args=args)
        node = ArmIKController()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n用户中断')
    except Exception as e:
        print(f'错误: {e}')
    finally:
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
