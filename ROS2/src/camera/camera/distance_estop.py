"""Pure state machine for the latched camera minimum-distance stop."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import time
from typing import Any


RESET_SOURCES_FIELD = 'reset_sources'
PERSON_CLASS_NAMES = frozenset({'person', 'preson'})
ARM_CLASS_NAMES = frozenset({'arm', 'robot_arm', 'robotarm'})
DEFAULT_DISTANCE_HISTORY_SIZE = 5


@dataclass(frozen=True)
class ResetIntent:
    """Request asking a named source to evaluate its own reset gate."""

    requester_source: str
    reason: str = ''


@dataclass(frozen=True)
class ResetDecision:
    """Result of checking a reset against current camera evidence."""

    accepted: bool
    cleared: bool
    reason: str


def resolve_distance_class_ids(
    names: Any,
) -> tuple[int, frozenset[int]]:
    """Resolve person and arm class ids from YOLO model metadata."""
    if isinstance(names, dict):
        items = names.items()
    elif isinstance(names, (list, tuple)):
        items = enumerate(names)
    else:
        raise ValueError('YOLO model names must be a dict, list, or tuple')

    person_ids: set[int] = set()
    arm_ids: set[int] = set()
    for raw_id, raw_name in items:
        if isinstance(raw_id, bool):
            raise ValueError('YOLO model class ids must be integers')
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError('YOLO model class ids must be integers') from exc
        normalized_name = str(raw_name).strip().lower().replace('-', '_').replace(' ', '_')
        if normalized_name in PERSON_CLASS_NAMES:
            person_ids.add(class_id)
        if normalized_name in ARM_CLASS_NAMES:
            arm_ids.add(class_id)

    if len(person_ids) != 1:
        raise ValueError(
            'YOLO model must define exactly one person class named '
            f'{sorted(PERSON_CLASS_NAMES)}; found ids={sorted(person_ids)}'
        )
    if not arm_ids:
        raise ValueError(
            'YOLO model must define at least one arm class named '
            f'{sorted(ARM_CLASS_NAMES)}'
        )
    person_class_id = next(iter(person_ids))
    if person_class_id in arm_ids:
        raise ValueError('YOLO person and arm class ids must not overlap')
    return person_class_id, frozenset(arm_ids)


def build_distance_estop_payload(
    *,
    source: str,
    active: bool,
    reason: str,
    threshold_m: float,
    distance_m: Any = None,
    trigger_distance_m: Any = None,
    release_distance_m: Any = None,
) -> dict[str, Any]:
    """Build the source-aware request emitted by the camera safety node."""
    payload: dict[str, Any] = {
        'source': str(source or 'min_distance_camera'),
        'active': bool(active),
        'latch': bool(active),
        'reason': str(reason or ''),
        'threshold_m': float(threshold_m),
    }
    distance = _finite_non_negative(distance_m)
    if distance is not None:
        payload['distance_m'] = distance
    trigger_distance = _finite_non_negative(trigger_distance_m)
    if trigger_distance is not None:
        payload['trigger_distance_m'] = trigger_distance
    release_distance = _finite_non_negative(release_distance_m)
    if release_distance is not None:
        payload['release_distance_m'] = release_distance
    return payload


def parse_reset_intent(
    text: str,
    target_source: str,
) -> tuple[ResetIntent | None, str | None]:
    """Parse a reset addressed to ``target_source`` from the request bus."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f'invalid JSON: {exc}'

    if not isinstance(payload, dict):
        return None, 'payload must be a JSON object'

    reset_sources = payload.get(RESET_SOURCES_FIELD)
    if reset_sources is None:
        return None, None
    if (
        not isinstance(reset_sources, list)
        or not all(
            isinstance(source, str) and source.strip()
            for source in reset_sources
        )
    ):
        return (
            None,
            f"field '{RESET_SOURCES_FIELD}' must be a list "
            'of non-empty strings',
        )
    if target_source not in {source.strip() for source in reset_sources}:
        return None, None

    source = payload.get('source')
    if not isinstance(source, str) or not source.strip():
        return None, "field 'source' must be a non-empty string"
    if payload.get('active') is not False:
        return None, 'camera reset intent requires active=false'
    if payload.get('latch') is not False:
        return None, 'camera reset intent requires latch=false'

    reason = payload.get('reason', '')
    return (
        ResetIntent(
            requester_source=source.strip(),
            reason='' if reason is None else str(reason).strip(),
        ),
        None,
    )


class LatchedDistanceEstop:
    """Latch on unsafe distance until an explicit safe reset."""

    def __init__(
        self,
        *,
        release_margin_m: float = 0.05,
        required_safe_frames: int = 3,
        max_evidence_age_seconds: float = 0.5,
        no_distance_release_seconds: float = 5.0,
        distance_history_size: int = DEFAULT_DISTANCE_HISTORY_SIZE,
    ) -> None:
        """Initialize configurable release-gate thresholds."""
        if not math.isfinite(release_margin_m) or release_margin_m < 0:
            raise ValueError(
                'release_margin_m must be a non-negative finite number'
            )
        if (
            isinstance(required_safe_frames, bool)
            or not isinstance(required_safe_frames, int)
            or required_safe_frames < 1
        ):
            raise ValueError('required_safe_frames must be at least 1')
        if (
            isinstance(distance_history_size, bool)
            or not isinstance(distance_history_size, int)
            or distance_history_size < required_safe_frames
            or distance_history_size > DEFAULT_DISTANCE_HISTORY_SIZE
        ):
            raise ValueError(
                'distance_history_size must be an integer between '
                'required_safe_frames and '
                f'{DEFAULT_DISTANCE_HISTORY_SIZE}'
            )
        if (
            not math.isfinite(max_evidence_age_seconds)
            or max_evidence_age_seconds <= 0
        ):
            raise ValueError('max_evidence_age_seconds must be greater than 0')
        if (
            not math.isfinite(no_distance_release_seconds)
            or no_distance_release_seconds <= 0
        ):
            raise ValueError(
                'no_distance_release_seconds must be greater than 0'
            )

        self.release_margin_m = float(release_margin_m)
        self.required_safe_frames = int(required_safe_frames)
        self.max_evidence_age_seconds = float(max_evidence_age_seconds)
        self.no_distance_release_seconds = float(no_distance_release_seconds)
        self.distance_history_size = int(distance_history_size)
        self.latched = False
        self._distance_history_m: deque[float] = deque(
            maxlen=self.distance_history_size
        )
        self.latest_distance_m: float | None = None
        self.trigger_distance_m: float | None = None
        self.latest_observation_monotonic: float | None = None
        self.latest_observation_valid = False
        self.latest_threshold_m: float | None = None
        self.no_distance_since_monotonic: float | None = None
        self.latest_no_distance_observation_monotonic: float | None = None

    @property
    def distance_history_m(self) -> tuple[float, ...]:
        """Return the retained valid distances from oldest to newest."""
        return tuple(self._distance_history_m)

    def observe_distance(
        self,
        distance_m: Any,
        stop_threshold_m: Any,
        *,
        now: float | None = None,
    ) -> bool:
        """Update one frame and return the effective camera stop level."""
        observed_at = time.monotonic() if now is None else float(now)
        threshold = _finite_non_negative(stop_threshold_m)
        distance = _finite_non_negative(distance_m)
        if threshold is None:
            self.invalidate_safety_evidence(now=observed_at)
            return self.latched

        if (
            self.latest_threshold_m is not None
            and not math.isclose(
                threshold,
                self.latest_threshold_m,
                abs_tol=1e-9,
            )
        ):
            self._distance_history_m.clear()
            self._invalidate_no_distance_evidence()

        self.latest_threshold_m = threshold
        if distance is None:
            self._observe_no_distance(observed_at)
            self.latest_distance_m = None
            self.latest_observation_monotonic = observed_at
            self.latest_observation_valid = False
            return self.latched

        self.latest_distance_m = distance
        self.latest_observation_monotonic = observed_at
        self.latest_observation_valid = True
        self._invalidate_no_distance_evidence()
        self._distance_history_m.append(distance)

        if distance < threshold:
            if not self.latched:
                self.trigger_distance_m = distance
            self.latched = True

        return self.latched

    def invalidate_safety_evidence(
        self,
        *,
        now: float | None = None,
        clear_distance_history: bool = False,
    ) -> None:
        """Mark sensor evidence invalid without treating failure as safety."""
        self.latest_distance_m = None
        self.latest_observation_monotonic = (
            time.monotonic() if now is None else float(now)
        )
        self.latest_observation_valid = False
        self._invalidate_no_distance_evidence()
        if clear_distance_history:
            self._distance_history_m.clear()

    def release_gate_m(self, stop_threshold_m: Any) -> float | None:
        """Return the distance required for a safe reset."""
        threshold = _finite_non_negative(stop_threshold_m)
        if threshold is None:
            return None
        return threshold + self.release_margin_m

    def request_reset(
        self,
        stop_threshold_m: Any,
        *,
        now: float | None = None,
    ) -> ResetDecision:
        """Clear only when current evidence satisfies the release gate."""
        if not self.latched:
            return ResetDecision(
                True,
                False,
                'camera emergency stop is already clear',
            )

        threshold = _finite_non_negative(stop_threshold_m)
        if threshold is None:
            return ResetDecision(False, False, 'stop threshold is invalid')
        requested_at = time.monotonic() if now is None else float(now)
        no_distance_duration = self._current_no_distance_duration(requested_at)
        if (
            no_distance_duration is not None
            and no_distance_duration >= self.no_distance_release_seconds
        ):
            self._clear_latch()
            return ResetDecision(
                True,
                True,
                f'no valid person-arm distance for '
                f'{no_distance_duration:.3f}s',
            )
        if (
            self.latest_threshold_m is None
            or not math.isclose(
                threshold,
                self.latest_threshold_m,
                abs_tol=1e-9,
            )
        ):
            self._distance_history_m.clear()
            return ResetDecision(
                False,
                False,
                'stop threshold changed; collect new distance readings',
            )

        release_distance = self.release_gate_m(threshold)
        assert release_distance is not None
        if (
            self.latest_observation_valid
            and self.latest_distance_m is not None
            and self.latest_distance_m < release_distance
        ):
            return ResetDecision(
                False,
                False,
                f'distance {self.latest_distance_m:.3f}m is below '
                f'release gate {release_distance:.3f}m',
            )
        recent_distances = self.distance_history_m[-self.required_safe_frames:]
        if len(recent_distances) < self.required_safe_frames:
            return ResetDecision(
                False,
                False,
                f'only {len(recent_distances)}/'
                f'{self.required_safe_frames} valid distance readings stored',
            )
        unsafe_recent_distances = tuple(
            distance
            for distance in recent_distances
            if distance < release_distance
        )
        if unsafe_recent_distances:
            return ResetDecision(
                False,
                False,
                f'recent distance {min(unsafe_recent_distances):.3f}m is '
                f'below release gate {release_distance:.3f}m',
            )

        self._clear_latch()
        return ResetDecision(True, True, 'camera release gate satisfied')

    def _observe_no_distance(self, observed_at: float) -> None:
        previous_observation = self.latest_no_distance_observation_monotonic
        self.latest_no_distance_observation_monotonic = observed_at
        if (
            self.no_distance_since_monotonic is None
            or previous_observation is None
            or observed_at < previous_observation
            or observed_at - previous_observation
            > self.max_evidence_age_seconds
        ):
            self.no_distance_since_monotonic = observed_at

    def _invalidate_no_distance_evidence(self) -> None:
        self.no_distance_since_monotonic = None
        self.latest_no_distance_observation_monotonic = None

    def _current_no_distance_duration(self, now: float) -> float | None:
        if (
            self.no_distance_since_monotonic is None
            or self.latest_no_distance_observation_monotonic is None
        ):
            return None
        evidence_age = now - self.latest_no_distance_observation_monotonic
        duration = now - self.no_distance_since_monotonic
        if (
            evidence_age < 0
            or evidence_age > self.max_evidence_age_seconds
            or duration < 0
        ):
            return None
        return duration

    def _clear_latch(self) -> None:
        self.latched = False
        self.trigger_distance_m = None
        self._distance_history_m.clear()
        self._invalidate_no_distance_evidence()


def _finite_non_negative(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return number
