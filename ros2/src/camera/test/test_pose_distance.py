import math

from camera.pose_distance import (
    ARM_ROLE,
    compute_min_distance_between_sets,
    conservative_distance_m,
    DepthMeasurement,
    DepthQuality,
    DepthRecoveryConfig,
    DetectionInstanceAssociator,
    map_pixel_between_image_shapes,
    measure_depth_at_keypoint,
    PERSON_ROLE,
    PoseDepthResolver,
    PoseKeypoint,
    PosePoint3D,
    select_keypoints_by_class,
    select_pose_keypoints_by_class,
)
import numpy as np


def _keypoint(
    role=PERSON_ROLE,
    detection_id=1,
    keypoint_id=0,
    pixel=(3, 3),
):
    return PoseKeypoint(
        role=role,
        class_id=0 if role == PERSON_ROLE else 1,
        detection_id=detection_id,
        keypoint_id=keypoint_id,
        pixel=pixel,
    )


def test_select_keypoints_uses_class_metadata_counts_and_confidence():
    keypoint_xy = np.zeros((2, 17, 2), dtype=np.float32)
    keypoint_confidence = np.zeros((2, 17), dtype=np.float32)

    keypoint_xy[0, 0] = (10.2, 20.7)
    keypoint_confidence[0, 0] = 0.9
    keypoint_xy[0, 1] = (30.0, 40.0)
    keypoint_confidence[0, 1] = 0.2

    keypoint_xy[1, 0] = (50.0, 60.0)
    keypoint_confidence[1, 0] = 0.8
    keypoint_xy[1, 5] = (70.0, 80.0)
    keypoint_confidence[1, 5] = 0.7
    keypoint_xy[1, 6] = (90.0, 100.0)
    keypoint_confidence[1, 6] = 0.99

    person_keypoints, arm_keypoints = select_keypoints_by_class(
        class_ids=np.array([0, 1]),
        keypoint_xy=keypoint_xy,
        keypoint_confidence=keypoint_confidence,
        keypoint_counts_by_class={0: 17, 1: 6},
        person_class_id=0,
        arm_class_ids={1},
        confidence_threshold=0.25,
    )

    assert person_keypoints == [(10, 21)]
    assert arm_keypoints == [(50, 60), (70, 80)]


def test_select_keypoints_rejects_invalid_coordinates():
    keypoint_xy = np.array(
        [[
            (0.0, 2.0),
            (2.0, 0.0),
            (-1.0, 2.0),
            (2.0, -1.0),
            (np.nan, 2.0),
            (10.0, 4.0),
            (3.0, 8.0),
            (3.0, 4.0),
        ]],
        dtype=np.float32,
    )

    person_keypoints, arm_keypoints = select_keypoints_by_class(
        class_ids=[0],
        keypoint_xy=keypoint_xy,
        keypoint_confidence=np.ones((1, 8), dtype=np.float32),
        keypoint_counts_by_class={0: 8},
        person_class_id=0,
        arm_class_ids={1},
        confidence_threshold=0.3,
        image_shape=(8, 10),
    )

    assert person_keypoints == [(3, 4)]
    assert arm_keypoints == []


def test_select_keypoints_rejects_confidence_below_good_version_default():
    person_keypoints, arm_keypoints = select_keypoints_by_class(
        class_ids=[0],
        keypoint_xy=np.array([[(3.0, 4.0)]], dtype=np.float32),
        keypoint_confidence=np.array([[0.25]], dtype=np.float32),
        keypoint_counts_by_class={0: 1},
        person_class_id=0,
        arm_class_ids={1},
        confidence_threshold=0.3,
        image_shape=(8, 10),
    )

    assert person_keypoints == []
    assert arm_keypoints == []


def test_select_pose_keypoints_retains_instance_and_keypoint_identity():
    keypoint_xy = np.array(
        [
            [(10.0, 20.0), (30.0, 40.0)],
            [(50.0, 60.0), (70.0, 80.0)],
        ],
        dtype=np.float32,
    )

    person_keypoints, arm_keypoints = select_pose_keypoints_by_class(
        class_ids=[0, 1],
        keypoint_xy=keypoint_xy,
        keypoint_confidence=np.ones((2, 2), dtype=np.float32),
        keypoint_counts_by_class={0: 2, 1: 2},
        person_class_id=0,
        arm_class_ids={1},
        confidence_threshold=0.25,
        detection_instance_ids=[41, 73],
    )

    assert person_keypoints[1] == PoseKeypoint(
        role=PERSON_ROLE,
        class_id=0,
        detection_id=41,
        keypoint_id=1,
        pixel=(30, 40),
    )
    assert arm_keypoints[0] == PoseKeypoint(
        role=ARM_ROLE,
        class_id=1,
        detection_id=73,
        keypoint_id=0,
        pixel=(50, 60),
    )


def test_detection_association_is_class_spatial_and_adjacent_frame_scoped():
    associator = DetectionInstanceAssociator(max_age_seconds=0.15)

    first_ids = associator.assign(
        class_ids=[0],
        boxes_xyxy=[[0.0, 0.0, 100.0, 100.0]],
        timestamp_seconds=1.0,
    )
    second_ids = associator.assign(
        class_ids=[0, 1],
        boxes_xyxy=[
            [5.0, 5.0, 105.0, 105.0],
            [0.0, 0.0, 100.0, 100.0],
        ],
        timestamp_seconds=1.1,
    )
    far_ids = associator.assign(
        class_ids=[0],
        boxes_xyxy=[[500.0, 500.0, 600.0, 600.0]],
        timestamp_seconds=1.2,
    )
    expired_ids = associator.assign(
        class_ids=[0],
        boxes_xyxy=[[505.0, 505.0, 605.0, 605.0]],
        timestamp_seconds=1.4,
    )

    assert second_ids[0] == first_ids[0]
    assert second_ids[1] != first_ids[0]
    assert far_ids[0] not in second_ids
    assert expired_ids[0] != far_ids[0]


def test_depth_measurement_quality_and_configured_uncertainty():
    config = DepthRecoveryConfig(
        direct_uncertainty_mm=12.0,
        recovered_uncertainty_mm=34.0,
    )
    direct_image = np.zeros((7, 7), dtype=np.float32)
    direct_image[3, 3] = 1000.0
    direct = measure_depth_at_keypoint(
        direct_image,
        (3, 3),
        reference_depth_mm=None,
        config=config,
    )

    recovery_image = np.full((7, 7), 2500.0, dtype=np.float32)
    recovery_image[3, 3] = 0.0
    recovery_image[2, 3] = 990.0
    recovery_image[3, 2] = 1000.0
    recovery_image[3, 4] = 1010.0
    recovered = measure_depth_at_keypoint(
        recovery_image,
        (3, 3),
        reference_depth_mm=1000.0,
        config=config,
    )
    invalid = measure_depth_at_keypoint(
        recovery_image,
        (3, 3),
        reference_depth_mm=None,
        config=config,
    )

    assert direct.quality == DepthQuality.DIRECT
    assert direct.uncertainty_mm == 12.0
    assert recovered.quality == DepthQuality.RECOVERED
    assert recovered.uncertainty_mm == 34.0
    assert recovered.depth_mm == 1000.0
    assert invalid.quality == DepthQuality.INVALID
    assert math.isinf(invalid.uncertainty_mm)


def test_depth_measurement_rejects_out_of_bounds_instead_of_sampling_border():
    depth_image = np.zeros((5, 5), dtype=np.float32)
    depth_image[0, 4] = 1000.0

    measurement = measure_depth_at_keypoint(
        depth_image,
        (99, -1),
        reference_depth_mm=None,
    )

    assert measurement.quality == DepthQuality.INVALID


def test_maps_1280x720_color_pixel_to_848x480_depth_pixel():
    assert map_pixel_between_image_shapes(
        (640, 360),
        (720, 1280, 3),
        (480, 848),
    ) == (424, 240)
    assert map_pixel_between_image_shapes(
        (1279, 719),
        (720, 1280, 3),
        (480, 848),
    ) == (847, 479)


def test_depth_resolver_samples_mapped_pixel_and_retains_color_keypoint():
    resolver = PoseDepthResolver(DepthRecoveryConfig())
    depth_image = np.zeros((480, 848), dtype=np.float32)
    depth_image[240, 424] = 1000.0
    color_keypoint = _keypoint(pixel=(640, 360))

    resolved = resolver.resolve(
        depth_image,
        [color_keypoint],
        1.0,
        source_image_shape=(720, 1280, 3),
    )

    assert resolved[0][0] is color_keypoint
    assert resolved[0][1].quality == DepthQuality.DIRECT
    assert resolved[0][1].depth_mm == 1000.0


def test_recovery_ignores_background_majority_outside_reference_gate():
    depth_image = np.full((7, 7), 2500.0, dtype=np.float32)
    depth_image[3, 3] = 0.0
    depth_image[2, 3] = 990.0
    depth_image[3, 2] = 1000.0
    depth_image[3, 4] = 1010.0

    measurement = measure_depth_at_keypoint(
        depth_image,
        (3, 3),
        reference_depth_mm=1000.0,
    )

    assert measurement.quality == DepthQuality.RECOVERED
    assert measurement.depth_mm == 1000.0
    assert measurement.recovery_window_size == 3


def test_recovery_expands_to_smallest_window_with_enough_candidates():
    depth_image = np.zeros((7, 7), dtype=np.float32)
    depth_image[1, 3] = 990.0
    depth_image[3, 1] = 1000.0
    depth_image[3, 5] = 1010.0

    measurement = measure_depth_at_keypoint(
        depth_image,
        (3, 3),
        reference_depth_mm=1000.0,
    )

    assert measurement.quality == DepthQuality.RECOVERED
    assert measurement.recovery_window_size == 5


def test_recovery_rejects_too_few_candidates():
    depth_image = np.full((7, 7), 2500.0, dtype=np.float32)
    depth_image[3, 3] = 0.0
    depth_image[2, 3] = 990.0
    depth_image[3, 2] = 1010.0

    measurement = measure_depth_at_keypoint(
        depth_image,
        (3, 3),
        reference_depth_mm=1000.0,
    )

    assert measurement.quality == DepthQuality.INVALID


def test_recovery_rejects_multimodal_candidates_with_large_mad():
    depth_image = np.zeros((7, 7), dtype=np.float32)
    neighbor_pixels = [
        (2, 2),
        (3, 2),
        (4, 2),
        (2, 3),
        (4, 3),
        (2, 4),
        (3, 4),
        (4, 4),
    ]
    for pixel in neighbor_pixels[:4]:
        depth_image[pixel[1], pixel[0]] = 960.0
    for pixel in neighbor_pixels[4:]:
        depth_image[pixel[1], pixel[0]] = 1040.0

    measurement = measure_depth_at_keypoint(
        depth_image,
        (3, 3),
        reference_depth_mm=1000.0,
    )

    assert measurement.quality == DepthQuality.INVALID


def test_depth_history_is_instance_isolated_and_expires_after_150_ms():
    resolver = PoseDepthResolver(DepthRecoveryConfig())
    direct_image = np.zeros((7, 7), dtype=np.float32)
    direct_image[3, 3] = 1000.0
    same_keypoint = _keypoint(detection_id=1)
    initial = resolver.resolve(direct_image, [same_keypoint], 1.0)
    assert initial[0][1].quality == DepthQuality.DIRECT

    recovery_image = np.full((7, 7), 2500.0, dtype=np.float32)
    recovery_image[3, 3] = 0.0
    recovery_image[2, 3] = 990.0
    recovery_image[3, 2] = 1000.0
    recovery_image[3, 4] = 1010.0

    recovered = resolver.resolve(recovery_image, [same_keypoint], 1.1)
    other_instance = resolver.resolve(
        recovery_image,
        [_keypoint(detection_id=2)],
        1.11,
    )
    expired = resolver.resolve(recovery_image, [same_keypoint], 1.3)

    assert recovered[0][1].quality == DepthQuality.RECOVERED
    assert other_instance[0][1].quality == DepthQuality.INVALID
    assert expired[0][1].quality == DepthQuality.INVALID


def test_same_keypoint_history_takes_priority_over_skeleton_reference():
    resolver = PoseDepthResolver(DepthRecoveryConfig())
    history_image = np.zeros((9, 9), dtype=np.float32)
    history_image[2, 2] = 1000.0
    history_keypoint = _keypoint(keypoint_id=0, pixel=(2, 2))
    resolver.resolve(history_image, [history_keypoint], 1.0)

    current_image = np.full((9, 9), 3000.0, dtype=np.float32)
    current_image[2, 2] = 0.0
    current_image[1, 2] = 990.0
    current_image[2, 1] = 1000.0
    current_image[2, 3] = 1010.0
    current_image[7, 7] = 2000.0
    resolved = resolver.resolve(
        current_image,
        [history_keypoint, _keypoint(keypoint_id=1, pixel=(7, 7))],
        1.1,
    )

    assert resolved[0][1].quality == DepthQuality.RECOVERED
    assert resolved[0][1].depth_mm == 1000.0


def test_same_instance_skeleton_neighbor_provides_recovery_reference():
    resolver = PoseDepthResolver(DepthRecoveryConfig())
    depth_image = np.full((9, 9), 2500.0, dtype=np.float32)
    depth_image[2, 2] = 0.0
    depth_image[1, 2] = 990.0
    depth_image[2, 1] = 1000.0
    depth_image[2, 3] = 1010.0
    depth_image[7, 7] = 1000.0
    keypoints = [
        _keypoint(keypoint_id=0, pixel=(2, 2)),
        _keypoint(keypoint_id=1, pixel=(7, 7)),
    ]

    resolved = resolver.resolve(depth_image, keypoints, 1.0)

    assert resolved[0][1].quality == DepthQuality.RECOVERED
    assert resolved[1][1].quality == DepthQuality.DIRECT


def test_skeleton_reference_does_not_cross_detection_instances():
    resolver = PoseDepthResolver(DepthRecoveryConfig())
    depth_image = np.full((9, 9), 2500.0, dtype=np.float32)
    depth_image[2, 2] = 0.0
    depth_image[1, 2] = 990.0
    depth_image[2, 1] = 1000.0
    depth_image[2, 3] = 1010.0
    depth_image[7, 7] = 1000.0
    keypoints = [
        _keypoint(detection_id=1, keypoint_id=0, pixel=(2, 2)),
        _keypoint(detection_id=2, keypoint_id=1, pixel=(7, 7)),
    ]

    resolved = resolver.resolve(depth_image, keypoints, 1.0)

    assert resolved[0][1].quality == DepthQuality.INVALID
    assert resolved[1][1].quality == DepthQuality.DIRECT


def test_arm_reference_uses_j1_to_j6_chain_neighbors():
    resolver = PoseDepthResolver(DepthRecoveryConfig())
    depth_image = np.full((9, 9), 2500.0, dtype=np.float32)
    depth_image[2, 2] = 0.0
    depth_image[1, 2] = 990.0
    depth_image[2, 1] = 1000.0
    depth_image[2, 3] = 1010.0
    depth_image[7, 7] = 1000.0
    keypoints = [
        _keypoint(role=ARM_ROLE, keypoint_id=4, pixel=(2, 2)),
        _keypoint(role=ARM_ROLE, keypoint_id=5, pixel=(7, 7)),
    ]

    resolved = resolver.resolve(depth_image, keypoints, 1.0)

    assert resolved[0][1].quality == DepthQuality.RECOVERED
    assert resolved[1][1].quality == DepthQuality.DIRECT


def test_compute_min_distance_between_keypoint_sets():
    person_points = [
        ((10, 10), (0.0, 0.0, 0.0)),
        ((20, 20), (1.0, 0.0, 0.0)),
    ]
    arm_points = [
        ((30, 30), (1.25, 0.0, 0.0)),
        ((40, 40), (3.0, 0.0, 0.0)),
    ]

    distance, person, arm = compute_min_distance_between_sets(
        person_points,
        arm_points,
    )

    assert math.isclose(distance, 0.25)
    assert person == person_points[1]
    assert arm == arm_points[0]


def test_compute_min_distance_returns_unavailable_for_empty_set():
    distance, person, arm = compute_min_distance_between_sets([], [])

    assert math.isinf(distance)
    assert person is None
    assert arm is None


def test_conservative_distance_subtracts_uncertainty_and_floors_at_zero():
    assert math.isclose(
        conservative_distance_m(
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            20.0,
            50.0,
        ),
        0.03,
    )
    assert conservative_distance_m(
        (0.0, 0.0, 0.0),
        (0.01, 0.0, 0.0),
        20.0,
        50.0,
    ) == 0.0
    assert conservative_distance_m(
        (0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0),
    ) == 0.1


def test_min_distance_reports_raw_euclidean_distance_despite_uncertainty():
    direct = DepthMeasurement(1000.0, DepthQuality.DIRECT, 20.0)
    recovered = DepthMeasurement(1000.0, DepthQuality.RECOVERED, 50.0)
    person = PosePoint3D(
        keypoint=_keypoint(),
        xyz_m=(0.0, 0.0, 0.0),
        measurement=direct,
    )
    arm = PosePoint3D(
        keypoint=_keypoint(role=ARM_ROLE),
        xyz_m=(0.1, 0.0, 0.0),
        measurement=recovered,
    )

    distance, closest_person, closest_arm = compute_min_distance_between_sets(
        [person],
        [arm],
    )

    assert math.isclose(distance, 0.1)
    assert closest_person is person
    assert closest_arm is arm
