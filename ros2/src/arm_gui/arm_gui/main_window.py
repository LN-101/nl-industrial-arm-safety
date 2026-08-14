from functools import partial
import math

from PyQt5 import QtCore, QtWidgets


JOINT_NAMES = [
    'j1_joint',
    'j2_joint',
    'j3_joint',
    'j4_joint',
    'j5_joint',
    'j6_joint',
]

JOINT_LIMITS_DEG = [
    (-180.0, 180.0),
    (-180.0, 0.0),
    (-180.0, 180.0),
    (-180.0, 180.0),
    (-126.0, 126.0),
    (-180.0, 180.0),
]

NEUTRAL_JOINTS_DEG = [0.0, -34.4, 45.8, 0.0, -28.6, 0.0]
SLIDER_SCALE = 10


class MainWindow(QtWidgets.QMainWindow):
    robot_joint_received = QtCore.pyqtSignal(object)
    mujoco_joint_received = QtCore.pyqtSignal(object)
    min_distance_received = QtCore.pyqtSignal(float)

    def __init__(self, ros_node, parent=None):
        super().__init__(parent)
        self.ros_node = ros_node
        self.joint_spinboxes = []
        self.joint_sliders = []
        self.joint_rows = {}
        self.command_joints_deg = list(NEUTRAL_JOINTS_DEG)
        self.robot_joints_deg = None
        self.mujoco_joints_deg = None
        self._syncing_joint_widgets = False

        self.auto_publish_timer = QtCore.QTimer(self)
        self.auto_publish_timer.setSingleShot(True)
        self.auto_publish_timer.timeout.connect(self.publish_joint_command)

        self.ros_node.robot_joint_callback = self.robot_joint_received.emit
        self.ros_node.mujoco_joint_callback = self.mujoco_joint_received.emit
        self.ros_node.min_distance_callback = self.min_distance_received.emit

        self.robot_joint_received.connect(
            partial(self.update_joint_feedback, 'robot')
        )
        self.mujoco_joint_received.connect(
            partial(self.update_joint_feedback, 'mujoco')
        )
        self.min_distance_received.connect(self.update_min_distance)

        self.setup_ui()
        self.apply_style()
        self.set_joint_widgets(NEUTRAL_JOINTS_DEG)
        self.statusBar().showMessage('GUI ready')

    def setup_ui(self):
        self.setWindowTitle('机械臂本地控制台')
        self.resize(1080, 720)

        root = QtWidgets.QWidget()
        root_layout = QtWidgets.QVBoxLayout(root)
        root_layout.setContentsMargins(18, 16, 18, 16)
        root_layout.setSpacing(14)
        self.setCentralWidget(root)

        header_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel('机械臂本地控制台')
        title.setObjectName('titleLabel')
        header_layout.addWidget(title)
        header_layout.addStretch(1)

        self.stop_button = QtWidgets.QPushButton('急停')
        self.stop_button.setCheckable(True)
        self.stop_button.setObjectName('stopButton')
        self.stop_button.clicked.connect(self.toggle_emergency_stop)
        header_layout.addWidget(self.stop_button)
        root_layout.addLayout(header_layout)

        content_layout = QtWidgets.QHBoxLayout()
        content_layout.setSpacing(14)
        root_layout.addLayout(content_layout, 1)

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        content_layout.addWidget(left_panel, 3)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        content_layout.addWidget(right_panel, 2)

        left_layout.addWidget(self.create_goal_group())
        left_layout.addWidget(self.create_joint_group(), 1)
        right_layout.addWidget(self.create_feedback_group(), 1)
        right_layout.addWidget(self.create_topic_group())

    def create_goal_group(self):
        group = QtWidgets.QGroupBox('末端目标 /goal')
        layout = QtWidgets.QGridLayout(group)
        layout.setColumnStretch(1, 1)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(8)

        self.goal_spinboxes = {}
        goal_config = [
            ('x', 'X (m)', -0.3, 0.3, 0.25),
            ('y', 'Y (m)', -0.3, 0.3, 0.25),
            ('z', 'Z (m)', 0.05, 0.3, 0.25),
        ]
        for row, (key, label, minimum, maximum, value) in enumerate(goal_config):
            layout.addWidget(QtWidgets.QLabel(label), row, 0)
            spinbox = QtWidgets.QDoubleSpinBox()
            spinbox.setRange(minimum, maximum)
            spinbox.setDecimals(3)
            spinbox.setSingleStep(0.005)
            spinbox.setValue(value)
            spinbox.setSuffix(' m')
            layout.addWidget(spinbox, row, 1)
            self.goal_spinboxes[key] = spinbox

        button_layout = QtWidgets.QHBoxLayout()
        self.publish_goal_button = QtWidgets.QPushButton('发布目标')
        self.publish_goal_button.clicked.connect(self.publish_goal)
        button_layout.addWidget(self.publish_goal_button)

        preset_button = QtWidgets.QPushButton('默认点')
        preset_button.clicked.connect(self.set_default_goal)
        button_layout.addWidget(preset_button)
        layout.addLayout(button_layout, 3, 0, 1, 2)

        return group

    def create_joint_group(self):
        group = QtWidgets.QGroupBox('手动关节命令 /control')
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(10)

        form = QtWidgets.QGridLayout()
        form.setColumnStretch(1, 1)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        layout.addLayout(form)

        for index, joint_name in enumerate(JOINT_NAMES):
            lower, upper = JOINT_LIMITS_DEG[index]
            label = QtWidgets.QLabel(f'J{index + 1}')
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(
                int(lower * SLIDER_SCALE),
                int(upper * SLIDER_SCALE),
            )
            slider.setSingleStep(1)
            slider.setPageStep(50)

            spinbox = QtWidgets.QDoubleSpinBox()
            spinbox.setRange(lower, upper)
            spinbox.setDecimals(1)
            spinbox.setSingleStep(1.0)
            spinbox.setSuffix(' deg')
            spinbox.setMinimumWidth(118)

            slider.valueChanged.connect(partial(self.on_joint_slider_changed, index))
            spinbox.valueChanged.connect(partial(self.on_joint_spinbox_changed, index))

            self.joint_sliders.append(slider)
            self.joint_spinboxes.append(spinbox)
            self.joint_rows[joint_name] = index

            form.addWidget(label, index, 0)
            form.addWidget(slider, index, 1)
            form.addWidget(spinbox, index, 2)

        options_layout = QtWidgets.QHBoxLayout()
        self.auto_publish_checkbox = QtWidgets.QCheckBox('实时发送')
        options_layout.addWidget(self.auto_publish_checkbox)
        options_layout.addStretch(1)
        layout.addLayout(options_layout)

        button_layout = QtWidgets.QHBoxLayout()
        publish_button = QtWidgets.QPushButton('发布关节')
        publish_button.clicked.connect(self.publish_joint_command)
        button_layout.addWidget(publish_button)

        neutral_button = QtWidgets.QPushButton('中性姿态')
        neutral_button.clicked.connect(lambda: self.set_joint_widgets(NEUTRAL_JOINTS_DEG))
        button_layout.addWidget(neutral_button)

        sync_robot_button = QtWidgets.QPushButton('同步真机')
        sync_robot_button.clicked.connect(partial(self.sync_from_feedback, 'robot'))
        button_layout.addWidget(sync_robot_button)

        sync_sim_button = QtWidgets.QPushButton('同步仿真')
        sync_sim_button.clicked.connect(partial(self.sync_from_feedback, 'mujoco'))
        button_layout.addWidget(sync_sim_button)
        layout.addLayout(button_layout)

        return group

    def create_feedback_group(self):
        group = QtWidgets.QGroupBox('状态反馈')
        layout = QtWidgets.QVBoxLayout(group)

        self.feedback_table = QtWidgets.QTableWidget(len(JOINT_NAMES), 4)
        self.feedback_table.setHorizontalHeaderLabels(
            ['关节', '真机 deg', '仿真 deg', '命令 deg']
        )
        self.feedback_table.verticalHeader().setVisible(False)
        self.feedback_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.feedback_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.feedback_table.horizontalHeader().setStretchLastSection(True)
        self.feedback_table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )

        for row, joint_name in enumerate(JOINT_NAMES):
            self.feedback_table.setItem(row, 0, QtWidgets.QTableWidgetItem(joint_name))
            for column in range(1, 4):
                item = QtWidgets.QTableWidgetItem('--')
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.feedback_table.setItem(row, column, item)

        layout.addWidget(self.feedback_table, 1)

        self.robot_time_label = QtWidgets.QLabel('真机反馈: --')
        self.mujoco_time_label = QtWidgets.QLabel('仿真反馈: --')
        self.distance_label = QtWidgets.QLabel('最小距离: --')
        layout.addWidget(self.robot_time_label)
        layout.addWidget(self.mujoco_time_label)
        layout.addWidget(self.distance_label)

        return group

    def create_topic_group(self):
        group = QtWidgets.QGroupBox('话题')
        layout = QtWidgets.QFormLayout(group)
        layout.addRow('/goal', QtWidgets.QLabel('geometry_msgs/msg/Point'))
        layout.addRow('/control', QtWidgets.QLabel('sensor_msgs/msg/JointState'))
        layout.addRow('/emergency_stop', QtWidgets.QLabel('std_msgs/msg/Bool'))
        layout.addRow('/robot_joint_state', QtWidgets.QLabel('sensor_msgs/msg/JointState'))
        layout.addRow('/mujoco_joint_state', QtWidgets.QLabel('sensor_msgs/msg/JointState'))
        return group

    def apply_style(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: #f5f7fb;
            }
            QLabel#titleLabel {
                color: #172033;
                font-size: 24px;
                font-weight: 700;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #d7deea;
                border-radius: 8px;
                color: #172033;
                font-weight: 600;
                margin-top: 10px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QPushButton {
                background: #2457c5;
                border: 0;
                border-radius: 6px;
                color: #ffffff;
                min-height: 30px;
                padding: 5px 12px;
            }
            QPushButton:hover {
                background: #1f4eb4;
            }
            QPushButton:pressed {
                background: #173c8d;
            }
            QPushButton#stopButton {
                background: #c72222;
                font-weight: 700;
                min-width: 92px;
            }
            QPushButton#stopButton:checked {
                background: #7f1010;
            }
            QDoubleSpinBox {
                min-height: 28px;
            }
            QTableWidget {
                border: 1px solid #d7deea;
                gridline-color: #e5e9f1;
                selection-background-color: #dbe8ff;
            }
            QHeaderView::section {
                background: #eef2f7;
                border: 0;
                color: #34415a;
                font-weight: 600;
                padding: 6px;
            }
            """
        )

    def set_default_goal(self):
        self.goal_spinboxes['x'].setValue(0.25)
        self.goal_spinboxes['y'].setValue(0.25)
        self.goal_spinboxes['z'].setValue(0.25)

    def publish_goal(self):
        x = self.goal_spinboxes['x'].value()
        y = self.goal_spinboxes['y'].value()
        z = self.goal_spinboxes['z'].value()
        self.ros_node.publish_goal(x, y, z)
        self.statusBar().showMessage(
            f'已发布目标: x={x:.3f}, y={y:.3f}, z={z:.3f}',
            3000,
        )

    def on_joint_slider_changed(self, index, raw_value):
        if self._syncing_joint_widgets:
            return
        value = raw_value / SLIDER_SCALE
        self.joint_spinboxes[index].setValue(value)

    def on_joint_spinbox_changed(self, index, value):
        if self._syncing_joint_widgets:
            return
        self.command_joints_deg[index] = float(value)
        self.joint_sliders[index].setValue(int(round(value * SLIDER_SCALE)))
        self.update_command_table()
        if self.auto_publish_checkbox.isChecked():
            self.auto_publish_timer.start(120)

    def set_joint_widgets(self, joints_deg):
        self._syncing_joint_widgets = True
        try:
            for index, value in enumerate(joints_deg):
                lower, upper = JOINT_LIMITS_DEG[index]
                clamped = max(lower, min(upper, float(value)))
                self.command_joints_deg[index] = clamped
                self.joint_spinboxes[index].setValue(clamped)
                self.joint_sliders[index].setValue(int(round(clamped * SLIDER_SCALE)))
        finally:
            self._syncing_joint_widgets = False
        self.update_command_table()
        if self.auto_publish_checkbox.isChecked():
            self.auto_publish_timer.start(120)

    def publish_joint_command(self):
        joints_rad = [math.radians(value) for value in self.command_joints_deg]
        try:
            self.ros_node.publish_joint_command(joints_rad)
        except ValueError as exc:
            self.statusBar().showMessage(str(exc), 5000)
            return
        self.statusBar().showMessage('已发布手动关节命令', 3000)

    def toggle_emergency_stop(self, checked):
        self.ros_node.publish_emergency_stop(checked)
        self.stop_button.setText('解除急停' if checked else '急停')
        message = '急停已触发' if checked else '急停已解除'
        self.statusBar().showMessage(message, 3000)

    def sync_from_feedback(self, source):
        joints = self.robot_joints_deg if source == 'robot' else self.mujoco_joints_deg
        if joints is None:
            label = '真机' if source == 'robot' else '仿真'
            self.statusBar().showMessage(f'没有可同步的{label}反馈', 3000)
            return
        self.set_joint_widgets(joints)

    def update_joint_feedback(self, source, msg):
        joints_deg = self.extract_joint_degrees(msg)
        if joints_deg is None:
            return

        now = QtCore.QDateTime.currentDateTime().toString('HH:mm:ss.zzz')
        if source == 'robot':
            self.robot_joints_deg = joints_deg
            column = 1
            self.robot_time_label.setText(f'真机反馈: {now}')
        else:
            self.mujoco_joints_deg = joints_deg
            column = 2
            self.mujoco_time_label.setText(f'仿真反馈: {now}')

        for row, value in enumerate(joints_deg):
            self.feedback_table.item(row, column).setText(f'{value:.1f}')

    def extract_joint_degrees(self, msg):
        if len(msg.position) < len(JOINT_NAMES):
            return None

        if msg.name:
            name_to_position = dict(zip(msg.name, msg.position))
            try:
                positions = [name_to_position[name] for name in JOINT_NAMES]
            except KeyError:
                positions = list(msg.position[:len(JOINT_NAMES)])
        else:
            positions = list(msg.position[:len(JOINT_NAMES)])

        return [math.degrees(float(position)) for position in positions]

    def update_command_table(self):
        for row, value in enumerate(self.command_joints_deg):
            self.feedback_table.item(row, 3).setText(f'{value:.1f}')

    def update_min_distance(self, value):
        self.distance_label.setText(f'最小距离: {value:.3f} m')

    def closeEvent(self, event):
        self.auto_publish_timer.stop()
        super().closeEvent(event)
