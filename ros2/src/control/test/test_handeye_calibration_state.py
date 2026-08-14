from __future__ import annotations

import unittest

from control.handeye_calibration_state import CalibrationStateMachine


class CalibrationStateMachineTest(unittest.TestCase):
    def test_nine_point_fake_run_completes_without_stale_results(self):
        machine = CalibrationStateMachine()
        machine.start(0.0, 0.0)
        now = 0.0
        for seq in range(1, 10):
            machine.request_ik(seq, now, 1.0)
            self.assertFalse(
                machine.accept_ik_result(seq - 1, True, now, 1.0))
            self.assertTrue(machine.accept_ik_result(seq, True, now, 1.0))
            self.assertTrue(machine.arrived(now, 0.0))
            self.assertTrue(machine.begin_sampling(now, 1.0))
            self.assertTrue(machine.sample_complete())
            now += 0.1
        self.assertTrue(machine.begin_return(10, now, 1.0))
        machine.complete()

        self.assertEqual(machine.state, 'completed')

    def test_stale_ik_result_is_ignored(self):
        machine = CalibrationStateMachine()
        machine.start(0.0, 0.5)
        machine.request_ik(4, 0.5, 10.0)

        accepted = machine.accept_ik_result(3, True, 1.0, 20.0)

        self.assertFalse(accepted)
        self.assertEqual(machine.state, 'waiting_ik')

    def test_matching_ik_result_advances_to_arrival(self):
        machine = CalibrationStateMachine()
        machine.start(0.0, 0.5)
        machine.request_ik(4, 0.5, 10.0)

        accepted = machine.accept_ik_result(4, True, 1.0, 20.0)

        self.assertTrue(accepted)
        self.assertEqual(machine.state, 'waiting_arrival')

    def test_estop_clears_sequence_and_never_returns(self):
        machine = CalibrationStateMachine()
        machine.start(0.0, 0.5)
        machine.request_ik(1, 0.5, 10.0)

        machine.estop()

        self.assertEqual(machine.state, 'estopped')
        self.assertIsNone(machine.active_seq)
        self.assertFalse(machine.begin_return(2, 1.0, 10.0))

    def test_return_has_its_own_timeout(self):
        machine = CalibrationStateMachine()
        machine.start(0.0, 0.5)
        self.assertTrue(machine.begin_return(5, 1.0, 3.0))

        self.assertFalse(machine.timed_out(4.0))
        self.assertTrue(machine.timed_out(4.01))

    def test_sampling_state_sequence(self):
        machine = CalibrationStateMachine()
        machine.start(0.0, 0.0)
        machine.request_ik(1, 0.0, 1.0)
        machine.accept_ik_result(1, True, 0.1, 1.0)
        machine.arrived(0.2, 0.5)

        self.assertFalse(machine.begin_sampling(0.6, 2.0))
        self.assertTrue(machine.begin_sampling(0.7, 2.0))
        self.assertTrue(machine.sample_complete())
        self.assertEqual(machine.state, 'sampling')


if __name__ == '__main__':
    unittest.main()
