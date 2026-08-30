from __future__ import annotations

from dataclasses import dataclass


TERMINAL_STATES = {'idle', 'completed', 'failed', 'estopped'}


@dataclass
class CalibrationStateMachine:
    state: str = 'idle'
    active_seq: int | None = None
    deadline: float | None = None
    failure_reason: str | None = None

    def start(self, now, activation_delay):
        if self.state not in TERMINAL_STATES:
            raise RuntimeError('calibration is already active')
        self.state = 'activating'
        self.active_seq = None
        self.failure_reason = None
        self.deadline = now + activation_delay

    def activation_ready(self, now):
        return self.state == 'activating' and now >= self.deadline

    def request_ik(self, seq, now, timeout):
        if self.state not in {'activating', 'sampling'}:
            raise RuntimeError(f'cannot request IK from {self.state}')
        self.state = 'waiting_ik'
        self.active_seq = seq
        self.deadline = now + timeout

    def accept_ik_result(self, seq, success, now, arrival_timeout):
        if self.state != 'waiting_ik' or seq != self.active_seq:
            return False
        if not success:
            self.fail('ik_failed')
            return True
        self.state = 'waiting_arrival'
        self.deadline = now + arrival_timeout
        return True

    def arrived(self, now, settle_time):
        if self.state != 'waiting_arrival':
            return False
        self.state = 'settling'
        self.deadline = now + settle_time
        return True

    def begin_sampling(self, now, detection_timeout):
        if self.state != 'settling' or now < self.deadline:
            return False
        self.state = 'collecting'
        self.deadline = now + detection_timeout
        return True

    def sample_complete(self):
        if self.state != 'collecting':
            return False
        self.state = 'sampling'
        self.deadline = None
        return True

    def begin_return(self, seq, now, timeout):
        if self.state == 'estopped':
            return False
        self.state = 'returning'
        self.active_seq = seq
        self.deadline = now + timeout
        return True

    def complete(self):
        self.state = 'completed'
        self.active_seq = None
        self.deadline = None

    def fail(self, reason):
        self.state = 'failed'
        self.active_seq = None
        self.deadline = None
        self.failure_reason = reason

    def estop(self):
        self.state = 'estopped'
        self.active_seq = None
        self.deadline = None
        self.failure_reason = 'emergency_stop'

    def timed_out(self, now):
        return self.deadline is not None and now > self.deadline
