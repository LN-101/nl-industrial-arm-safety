"""Tests for the camera minimum-distance emergency-stop state machine."""

from __future__ import annotations

import json
import unittest

from camera.distance_estop import (
    build_distance_estop_payload,
    LatchedDistanceEstop,
    parse_reset_intent,
    resolve_distance_class_ids,
)


class LatchedDistanceEstopTest(unittest.TestCase):
    """Verify latch and safety-gated release behavior."""

    def test_unsafe_distance_stays_latched(self) -> None:
        """Safe or invalid frames must not clear a triggered latch."""
        state = LatchedDistanceEstop()

        self.assertTrue(state.observe_distance(0.19, 0.2, now=1.0))
        self.assertTrue(state.observe_distance(0.40, 0.2, now=1.1))
        self.assertTrue(state.observe_distance(None, 0.2, now=1.2))
        self.assertTrue(state.latched)
        self.assertEqual(state.distance_history_m, (0.19, 0.40))

    def test_trigger_distance_is_preserved_until_latch_clears(self) -> None:
        state = LatchedDistanceEstop(required_safe_frames=1)

        state.observe_distance(0.19, 0.2, now=1.0)
        state.observe_distance(0.10, 0.2, now=1.1)
        state.observe_distance(0.30, 0.2, now=1.2)

        self.assertEqual(state.trigger_distance_m, 0.19)
        self.assertTrue(state.request_reset(0.2, now=1.2).cleared)
        self.assertIsNone(state.trigger_distance_m)
        self.assertEqual(state.distance_history_m, ())

        state.observe_distance(0.15, 0.2, now=1.3)
        self.assertEqual(state.trigger_distance_m, 0.15)

    def test_reset_requires_three_safe_distance_readings(self) -> None:
        """Reset is allowed only when the latest three readings are safe."""
        state = LatchedDistanceEstop(
            release_margin_m=0.05,
            required_safe_frames=3,
        )
        state.observe_distance(0.19, 0.2, now=1.0)

        state.observe_distance(0.25, 0.2, now=1.1)
        self.assertFalse(state.request_reset(0.2, now=1.1).accepted)
        state.observe_distance(0.26, 0.2, now=1.2)
        self.assertFalse(state.request_reset(0.2, now=1.2).accepted)
        state.observe_distance(0.25, 0.2, now=1.3)

        decision = state.request_reset(0.2, now=1.3)

        self.assertTrue(decision.accepted)
        self.assertTrue(decision.cleared)
        self.assertFalse(state.latched)

    def test_distance_history_retains_only_latest_five_readings(self) -> None:
        state = LatchedDistanceEstop(distance_history_size=5)

        distances = (0.1, 0.25, 0.26, 0.27, 0.28, 0.29)
        for index, distance in enumerate(distances):
            state.observe_distance(distance, 0.2, now=1.0 + index * 0.1)

        self.assertEqual(
            state.distance_history_m,
            (0.25, 0.26, 0.27, 0.28, 0.29),
        )

    def test_invalid_distance_starts_timer_without_entering_history(self) -> None:
        state = LatchedDistanceEstop()
        state.observe_distance(0.1, 0.2, now=1.0)
        state.observe_distance(0.3, 0.2, now=1.1)
        state.observe_distance(0.3, 0.2, now=1.2)

        state.observe_distance(float('inf'), 0.2, now=1.3)

        self.assertEqual(state.distance_history_m, (0.1, 0.3, 0.3))
        self.assertEqual(state.no_distance_since_monotonic, 1.3)
        decision = state.request_reset(0.2, now=1.3)
        self.assertFalse(decision.accepted)
        self.assertIn('below release gate', decision.reason)

    def test_safe_readings_persist_across_short_dropouts_and_long_intervals(self) -> None:
        """Stored readings are ordered by acquisition, not frame continuity."""
        state = LatchedDistanceEstop(
            release_margin_m=0.05,
            required_safe_frames=3,
            max_evidence_age_seconds=0.5,
        )
        state.observe_distance(0.1, 0.2, now=1.0)

        state.observe_distance(0.3, 0.2, now=1.1)
        state.observe_distance(None, 0.2, now=1.2)
        state.observe_distance(0.3, 0.2, now=2.0)
        state.observe_distance(None, 0.2, now=2.1)
        state.observe_distance(0.3, 0.2, now=3.0)

        decision = state.request_reset(0.2, now=3.0)

        self.assertTrue(decision.accepted)
        self.assertTrue(decision.cleared)
        self.assertFalse(state.latched)

    def test_only_latest_three_of_retained_five_control_release(self) -> None:
        state = LatchedDistanceEstop(
            release_margin_m=0.05,
            required_safe_frames=3,
        )
        state.observe_distance(0.1, 0.2, now=1.0)
        state.observe_distance(0.3, 0.2, now=1.1)
        state.observe_distance(0.3, 0.2, now=1.2)
        state.observe_distance(0.22, 0.2, now=1.3)
        state.observe_distance(0.3, 0.2, now=1.4)
        state.observe_distance(0.3, 0.2, now=1.5)

        self.assertEqual(
            state.distance_history_m,
            (0.3, 0.3, 0.22, 0.3, 0.3),
        )
        decision = state.request_reset(0.2, now=1.5)
        self.assertFalse(decision.accepted)
        self.assertIn('0.220m', decision.reason)

        state.observe_distance(0.3, 0.2, now=1.6)
        decision = state.request_reset(0.2, now=1.6)

        self.assertTrue(decision.accepted)
        self.assertTrue(decision.cleared)

    def test_unsafe_reading_retriggers_latch_and_blocks_recent_history(self) -> None:
        state = LatchedDistanceEstop(
            release_margin_m=0.05,
            required_safe_frames=3,
        )
        state.observe_distance(0.1, 0.2, now=1.0)
        state.observe_distance(0.3, 0.2, now=1.1)
        state.observe_distance(0.3, 0.2, now=1.2)

        state.observe_distance(0.15, 0.2, now=1.3)

        self.assertTrue(state.latched)
        self.assertFalse(state.request_reset(0.2, now=1.3).accepted)

        state.observe_distance(0.3, 0.2, now=1.4)
        state.observe_distance(0.3, 0.2, now=1.5)
        self.assertFalse(state.request_reset(0.2, now=1.5).accepted)
        state.observe_distance(0.3, 0.2, now=1.6)
        self.assertTrue(state.request_reset(0.2, now=1.6).cleared)

    def test_unsafe_distance_rejects_reset(self) -> None:
        """A current unsafe distance rejects an otherwise explicit reset."""
        state = LatchedDistanceEstop()
        state.observe_distance(0.1, 0.2, now=1.0)

        decision = state.request_reset(0.2, now=1.0)

        self.assertFalse(decision.accepted)
        self.assertIn('below release gate', decision.reason)

    def test_release_gate_adds_configured_margin(self) -> None:
        state = LatchedDistanceEstop(release_margin_m=0.05)

        self.assertEqual(state.release_gate_m(0.6), 0.65)
        self.assertIsNone(state.release_gate_m(float('nan')))

    def test_stored_safe_history_does_not_expire_by_wall_time(self) -> None:
        """The latest readings remain evidence until replaced or invalidated."""
        state = LatchedDistanceEstop(max_evidence_age_seconds=0.5)
        state.observe_distance(0.1, 0.2, now=1.0)
        state.observe_distance(0.3, 0.2, now=1.1)
        state.observe_distance(0.3, 0.2, now=2.0)
        state.observe_distance(0.3, 0.2, now=3.0)

        decision = state.request_reset(0.2, now=100.0)

        self.assertTrue(decision.accepted)
        self.assertTrue(decision.cleared)

    def test_five_seconds_without_valid_distance_allows_confirmed_reset(self) -> None:
        state = LatchedDistanceEstop(no_distance_release_seconds=5.0)
        state.observe_distance(0.1, 0.2, now=1.0)

        for frame in range(50):
            state.observe_distance(None, 0.2, now=2.0 + frame * 0.1)
        self.assertFalse(state.request_reset(0.2, now=6.9).accepted)
        state.observe_distance(None, 0.2, now=7.0)

        decision = state.request_reset(0.2, now=7.0)

        self.assertTrue(decision.accepted)
        self.assertTrue(decision.cleared)
        self.assertIn('no valid person-arm distance for 5.000s', decision.reason)

    def test_valid_distance_clears_no_distance_release_eligibility(self) -> None:
        state = LatchedDistanceEstop(no_distance_release_seconds=5.0)
        state.observe_distance(0.1, 0.2, now=1.0)
        for frame in range(51):
            state.observe_distance(None, 0.2, now=2.0 + frame * 0.1)

        state.observe_distance(0.3, 0.2, now=7.1)

        decision = state.request_reset(0.2, now=7.1)
        self.assertFalse(decision.accepted)
        self.assertIsNone(state.no_distance_since_monotonic)
        self.assertNotIn('no valid person-arm distance', decision.reason)

    def test_detection_callback_gap_restarts_no_distance_duration(self) -> None:
        state = LatchedDistanceEstop(
            max_evidence_age_seconds=0.5,
            no_distance_release_seconds=5.0,
        )
        state.observe_distance(0.1, 0.2, now=1.0)
        state.observe_distance(None, 0.2, now=2.0)

        state.observe_distance(None, 0.2, now=7.0)

        decision = state.request_reset(0.2, now=7.0)
        self.assertFalse(decision.accepted)

    def test_sensor_failure_clears_timer_but_keeps_distance_history(self) -> None:
        state = LatchedDistanceEstop(no_distance_release_seconds=5.0)
        state.observe_distance(0.1, 0.2, now=1.0)
        for frame in range(51):
            state.observe_distance(None, 0.2, now=2.0 + frame * 0.1)

        state.invalidate_safety_evidence(now=7.1)

        decision = state.request_reset(0.2, now=7.1)
        self.assertFalse(decision.accepted)
        self.assertIsNone(state.no_distance_since_monotonic)
        self.assertEqual(state.distance_history_m, (0.1,))

    def test_threshold_change_clears_old_distance_history(self) -> None:
        state = LatchedDistanceEstop()
        state.observe_distance(0.1, 0.2, now=1.0)
        state.observe_distance(0.3, 0.2, now=1.1)
        state.observe_distance(0.3, 0.2, now=1.2)

        state.observe_distance(0.35, 0.25, now=1.3)

        self.assertEqual(state.distance_history_m, (0.35,))
        decision = state.request_reset(0.25, now=1.3)
        self.assertFalse(decision.accepted)
        self.assertIn('only 1/3', decision.reason)

    def test_history_size_must_hold_required_safe_readings(self) -> None:
        invalid_configs = (
            {'required_safe_frames': 6, 'distance_history_size': 5},
            {'required_safe_frames': 3, 'distance_history_size': 6},
        )

        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    LatchedDistanceEstop(**config)


class DistanceClassIdsTest(unittest.TestCase):
    def test_resolves_current_model_misspelled_person_name(self) -> None:
        person_class, arm_classes = resolve_distance_class_ids(
            {0: 'preson', 1: 'arm'}
        )

        self.assertEqual(person_class, 0)
        self.assertEqual(arm_classes, frozenset({1}))

    def test_resolves_standard_names_from_sequence(self) -> None:
        person_class, arm_classes = resolve_distance_class_ids(
            ['person', 'robot-arm']
        )

        self.assertEqual(person_class, 0)
        self.assertEqual(arm_classes, frozenset({1}))

    def test_rejects_missing_or_ambiguous_safety_classes(self) -> None:
        invalid_names = (
            {0: 'arm'},
            {0: 'person'},
            {0: 'person', 1: 'preson', 2: 'arm'},
        )

        for names in invalid_names:
            with self.subTest(names=names):
                with self.assertRaises(ValueError):
                    resolve_distance_class_ids(names)


class ResetIntentParserTest(unittest.TestCase):
    """Verify reset intents are explicit and source-addressed."""

    def test_parse_accepts_explicit_reset_for_camera_source(self) -> None:
        """A confirmed inactive request can address the camera source."""
        intent, error = parse_reset_intent(
            json.dumps(
                {
                    'source': 'voice_assistant',
                    'active': False,
                    'latch': False,
                    'reason': '确认解除急停',
                    'reset_sources': ['min_distance_camera'],
                }
            ),
            'min_distance_camera',
        )

        self.assertIsNone(error)
        self.assertIsNotNone(intent)
        assert intent is not None
        self.assertEqual(intent.requester_source, 'voice_assistant')

    def test_parse_ignores_clear_without_explicit_camera_reset(self) -> None:
        """Clearing another source must not implicitly clear the camera."""
        intent, error = parse_reset_intent(
            '{"source":"voice_assistant","active":false,"latch":false}',
            'min_distance_camera',
        )

        self.assertIsNone(error)
        self.assertIsNone(intent)

    def test_parse_rejects_active_camera_reset_intent(self) -> None:
        """An active request cannot double as a reset intent."""
        intent, error = parse_reset_intent(
            '{"source":"voice_assistant","active":true,"latch":true,'
            '"reset_sources":["min_distance_camera"]}',
            'min_distance_camera',
        )

        self.assertIsNone(intent)
        self.assertIsNotNone(error)


class DistanceEstopPayloadTest(unittest.TestCase):
    """Verify camera request payload latch semantics."""

    def test_active_request_latches_and_clear_request_does_not(self) -> None:
        """Active camera requests latch; owned clear requests do not."""
        active = build_distance_estop_payload(
            source='min_distance_camera',
            active=True,
            reason='unsafe',
            threshold_m=0.2,
            distance_m=0.1,
            trigger_distance_m=0.12,
            release_distance_m=0.25,
        )
        clear = build_distance_estop_payload(
            source='min_distance_camera',
            active=False,
            reason='reset accepted',
            threshold_m=0.2,
            distance_m=0.3,
        )

        self.assertTrue(active['latch'])
        self.assertEqual(active['trigger_distance_m'], 0.12)
        self.assertEqual(active['distance_m'], 0.1)
        self.assertEqual(active['release_distance_m'], 0.25)
        self.assertFalse(clear['latch'])


if __name__ == '__main__':
    unittest.main()
