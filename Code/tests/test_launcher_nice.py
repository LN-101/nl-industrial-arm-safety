import subprocess
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
HELPER = ROOT_DIR / "launcher_nice.sh"
NO_ROS_LAUNCHER = ROOT_DIR / "start_web_no_ros.sh"
WITH_ROS_LAUNCHER = ROOT_DIR / "start_web_with_ros.sh"


class LauncherNiceTest(unittest.TestCase):
    def run_adjustment(
        self,
        requested: str,
        inherited: str,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; nice_adjustment "$2" "$3" AI_OV_TEST_NICE',
                "bash",
                str(HELPER),
                requested,
                inherited,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_adjustment_keeps_default_priority(self) -> None:
        result = self.run_adjustment("0", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0")

    def test_adjustment_demotes_to_ten(self) -> None:
        result = self.run_adjustment("10", "0")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "10")

        effective = subprocess.run(
            ["nice", "-n", result.stdout.strip(), "sh", "-c", "ps -o ni= -p $$"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(effective.stdout.strip(), "10")

    def test_adjustment_rejects_invalid_values(self) -> None:
        for value in ("invalid", "-1", "20"):
            with self.subTest(value=value):
                result = self.run_adjustment(value, "0")
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be an integer from 0 through 19", result.stderr)

    def test_adjustment_rejects_unattainable_priority_raise(self) -> None:
        result = self.run_adjustment("0", "5")
        self.assertEqual(result.returncode, 2)
        self.assertIn("launcher inherited effective nice 5", result.stderr)
        self.assertIn("cannot raise its priority", result.stderr)

    def run_cpu_weight_validation(self, value: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; validate_cpu_weight "$2" AI_OV_TEST_CPU_WEIGHT',
                "bash",
                str(HELPER),
                value,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_cpu_weight_accepts_valid_range(self) -> None:
        for value in ("1", "25", "100", "500", "10000"):
            with self.subTest(value=value):
                result = self.run_cpu_weight_validation(value)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_cpu_weight_rejects_invalid_values(self) -> None:
        # "08"/"099" guard against bash octal arithmetic errors evaluating as a
        # pass; the 20-digit value guards against intmax overflow in (( )).
        for value in ("0", "-5", "10001", "abc", "", "08", "099", "99999999999999999999"):
            with self.subTest(value=value):
                result = self.run_cpu_weight_validation(value)
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be an integer from 1 through 10000", result.stderr)

    def run_cpu_list_validation(self, value: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; validate_cpu_list "$2" AI_OV_TEST_CPUS',
                "bash",
                str(HELPER),
                value,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_cpu_list_accepts_valid_lists(self) -> None:
        for value in ("4-9", "0", "0,2,4", "4-7,10", "12,13"):
            with self.subTest(value=value):
                result = self.run_cpu_list_validation(value)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_cpu_list_rejects_invalid_lists(self) -> None:
        for value in ("", "4-", "-9", "4..9", "a", "4 9", "4,,9"):
            with self.subTest(value=value):
                result = self.run_cpu_list_validation(value)
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be a taskset CPU list", result.stderr)

    def test_launchers_reject_invalid_settings_before_runtime_checks(self) -> None:
        cases = (
            (NO_ROS_LAUNCHER, {"AI_OV_VOICE_NICE": "-1"}),
            (WITH_ROS_LAUNCHER, {"AI_OV_MIN_DIS_NICE": "20"}),
        )
        for launcher, environment in cases:
            with self.subTest(launcher=launcher.name, environment=environment):
                result = subprocess.run(
                    [str(launcher)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be an integer from 0 through 19", result.stderr)

    def test_launchers_reject_invalid_cpu_weight_before_runtime_checks(self) -> None:
        cases = (
            (NO_ROS_LAUNCHER, {"AI_OV_VOICE_CPU_WEIGHT": "0"}),
            (WITH_ROS_LAUNCHER, {"AI_OV_MIN_DIS_CPU_WEIGHT": "10001"}),
        )
        for launcher, environment in cases:
            with self.subTest(launcher=launcher.name, environment=environment):
                result = subprocess.run(
                    [str(launcher)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be an integer from 1 through 10000", result.stderr)

    def test_launchers_reject_invalid_cpu_list_before_runtime_checks(self) -> None:
        cases = (
            (NO_ROS_LAUNCHER, {"AI_OV_MIN_DIS_CPUS": "4-"}),
            (WITH_ROS_LAUNCHER, {"AI_OV_MIN_DIS_CPUS": "abc"}),
        )
        for launcher, environment in cases:
            with self.subTest(launcher=launcher.name, environment=environment):
                result = subprocess.run(
                    [str(launcher)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be a taskset CPU list", result.stderr)

    def test_launchers_reject_invalid_min_dis_cpu_threads_before_runtime_checks(self) -> None:
        for launcher in (NO_ROS_LAUNCHER, WITH_ROS_LAUNCHER):
            with self.subTest(launcher=launcher.name):
                result = subprocess.run(
                    [str(launcher)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env={"AI_OV_MIN_DIS_CPU_THREADS": "0"},
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("must be a positive integer", result.stderr)

    def test_launchers_default_to_low_latency_moss_buffer(self) -> None:
        expected_default = 'MOSS_PCM_BUFFER_SECONDS="${AI_OV_MOSS_PCM_BUFFER_SECONDS:-0.48}"'

        for launcher in (NO_ROS_LAUNCHER, WITH_ROS_LAUNCHER):
            with self.subTest(launcher=launcher.name):
                self.assertIn(expected_default, launcher.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
