# Offline regression tests for the 100-launch stress executor.
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("all_reduce_stress", ROOT / "stress.py")
stress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stress)
EMPTY_DMESG = {"time": "test", "return_code": 0, "text": "", "error": ""}


def executable(path, body):
    path.write_text(body)
    path.chmod(0o755)
    return path


class StressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_original_case_parameters(self):
        command = stress.mpi_command(Path("/test/all_reduce_perf"), "eth0", "mpirun")
        self.assertEqual(command[command.index("-np") + 1], "2")
        self.assertEqual(command[command.index("-npernode") + 1], "2")
        self.assertEqual(command[command.index("btl_tcp_if_include") + 1], "eth0")
        self.assertEqual(command[command.index("/test/all_reduce_perf") + 1:], [
            "-t", "1", "-g", "1", "-d", "half", "-b", "8", "-e", "16G",
            "-f", "2", "-n", "20", "-w", "5", "--elapsed_type", "1",
        ])
        self.assertEqual(command.count("-x"), 6)
        for name, value in stress.CASE_ENV.items():
            self.assertIn(f"{name}={value}", command)

    def launch(self, text, code=0):
        directory = self.root / "run"
        directory.mkdir()
        with patch.object(stress, "kernel_snapshot", return_value=EMPTY_DMESG):
            return stress.run_once([sys.executable, "-c", f"print({text!r}); raise SystemExit({code})"],
                                   directory, 3, 0.1)

    def test_qp_diagnostics_are_not_failures(self):
        record = self.launch("create qp 0 failed\nmax send wr 384 is not power of 2")
        self.assertEqual(record["status"], "PASSED")
        self.assertEqual(record["signatures"]["qp_create_warning"], 1)
        self.assertFalse(record["target_hggc_failure"])

    def test_original_incident_evidence(self):
        record = self.launch("Hg failure: cmd=hgMemMap(ptr, size) 'invalid argument'\n"
                             "Test PCCL failure common.cu:1248 'unhandled hggc error'", 3)
        self.assertEqual(record["status"], "FAILED")
        self.assertEqual(record["return_code"], 3)
        self.assertTrue(record["target_hggc_failure"])
        self.assertIn("hg_mem_map_invalid_argument", record["fatal_signatures"])
        self.assertTrue((self.root / "run/dmesg.before.log").exists())

    def test_fatal_text_with_zero_exit_is_not_false_pass(self):
        self.assertEqual(self.launch("unhandled hggc error")["status"], "FAILED")

    def test_nonzero_without_target_signature_is_still_failure(self):
        record = self.launch("unrelated failure", 7)
        self.assertEqual(record["status"], "FAILED")
        self.assertFalse(record["target_hggc_failure"])

    def test_dmesg_append_and_rollover(self):
        def snap(text):
            return dict(EMPTY_DMESG, text=text)
        self.assertEqual(stress.kernel_delta(snap("a\nb\n"), snap("a\nb\nc\n")), ("c\n", "appended"))
        self.assertEqual(stress.kernel_delta(snap("a\nb\nc\n"), snap("b\nc\nd\n")), ("d\n", "ring_buffer_rolled"))
        self.assertEqual(stress.kernel_delta(snap("a\n"), snap("z\n"))[1],
                         "no_overlap_full_after_not_an_exact_delta")
        self.assertEqual(stress.kernel_delta(dict(snap(""), return_code=1), snap("z\n")),
                         ("", "unavailable"))

    def test_dmesg_permission_error_does_not_abort_stress(self):
        with patch.object(stress.subprocess, "run", side_effect=FileNotFoundError("no dmesg")):
            result = stress.kernel_snapshot()
        self.assertIsNone(result["return_code"])
        self.assertIn("no dmesg", result["error"])

    def test_timeout_kills_remaining_local_rank(self):
        directory = self.root / "run"
        directory.mkdir()
        child_pid = self.root / "child.pid"
        child_code = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
        parent_code = (
            "import subprocess,sys,time; from pathlib import Path; "
            f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]); "
            f"Path({str(child_pid)!r}).write_text(str(p.pid)); time.sleep(60)"
        )
        with patch.object(stress, "kernel_snapshot", return_value=EMPTY_DMESG):
            record = stress.run_once([sys.executable, "-c", parent_code], directory, 0.3, 0.2)
        self.assertEqual(record["status"], "TIMEOUT")
        self.assertLess(record["duration_seconds"], 3)
        pid = int(child_pid.read_text())
        for _ in range(50):
            state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
            if not state or state.startswith("Z"):
                break
            time.sleep(0.02)
        self.assertTrue(not state or state.startswith("Z"), f"local rank {pid} still running: {state}")

    def test_launcher_not_found_is_recorded(self):
        directory = self.root / "run"
        directory.mkdir()
        with patch.object(stress, "kernel_snapshot", return_value=EMPTY_DMESG):
            result = stress.run_once([str(self.root / "missing")], directory, 1, 0.1)
        self.assertEqual(result["status"], "LAUNCH_ERROR")

    def test_100_launches_continue_after_failures_and_same_binary(self):
        binary = executable(self.root / "all_reduce_perf", "#!/bin/sh\necho unused\n")
        launcher = executable(self.root / "mpirun", f"#!{sys.executable}\n" +
                              "import os,sys\nfrom pathlib import Path\n"
                              "counter=Path(os.environ['FAKE_COUNT'])\n"
                              "n=int(counter.read_text())+1 if counter.exists() else 1\n"
                              "counter.write_text(str(n))\n"
                              "print('create qp 0 failed; max send wr 384 is not power of 2')\n"
                              "if n in (7,42,83):\n"
                              "    print(\"hgMemMap: invalid argument; unhandled hggc error\")\n"
                              "    sys.exit(3)\n")
        output = self.root / "results"
        with patch.dict(os.environ, {"FAKE_COUNT": str(self.root / "counter")}), \
                patch.object(stress, "kernel_snapshot", return_value=EMPTY_DMESG), \
                patch("builtins.print"):
            code = stress.main(["--binary", str(binary), "--mpirun", str(launcher),
                                "--output", str(output), "--runs", "100"])
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(code, 1)
        self.assertEqual((self.root / "counter").read_text(), "100")
        self.assertEqual(summary["completed_runs"], 100)
        self.assertEqual(summary["passed"], 97)
        self.assertEqual(summary["failed_iterations"], [7, 42, 83])
        self.assertEqual(summary["target_hggc_failure_iterations"], [7, 42, 83])
        self.assertEqual(summary["failure_rate_percent"], 3.0)
        self.assertFalse(summary["artifact_changed"])
        self.assertTrue(summary["complete"])
        self.assertEqual(len(list(output.glob("run-*/raw.log"))), 100)
        self.assertEqual(len((output / "summary.csv").read_text().splitlines()), 101)

    def test_existing_evidence_is_never_overwritten(self):
        binary = executable(self.root / "binary", "#!/bin/sh\nexit 0\n")
        output = self.root / "results"
        output.mkdir()
        (output / "evidence").write_text("preserve")
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            stress.main(["--binary", str(binary), "--mpirun", "/bin/sh", "--output", str(output)])
        self.assertEqual((output / "evidence").read_text(), "preserve")

    def test_artifact_change_is_detected(self):
        binary = executable(self.root / "binary", "#!/bin/sh\nexit 0\n")
        launcher = executable(self.root / "mpi", f"#!{sys.executable}\nfrom pathlib import Path\n"
                              f"Path({str(binary)!r}).write_text('changed')\n")
        with patch.object(stress, "kernel_snapshot", return_value=EMPTY_DMESG), patch("builtins.print"):
            code = stress.main(["--binary", str(binary), "--mpirun", str(launcher),
                                "--output", str(self.root / "results"), "--runs", "1"])
        self.assertEqual(code, 1)
        self.assertTrue(json.loads((self.root / "results/summary.json").read_text())["artifact_changed"])

    def test_interrupt_keeps_partial_summary(self):
        binary = executable(self.root / "binary", "#!/bin/sh\nexit 0\n")
        with patch.object(stress, "run_once", side_effect=KeyboardInterrupt), patch("builtins.print"):
            code = stress.main(["--binary", str(binary), "--mpirun", "/bin/sh",
                                "--output", str(self.root / "results")])
        summary = json.loads((self.root / "results/summary.json").read_text())
        self.assertEqual(code, 1)
        self.assertTrue(summary["interrupted"])
        self.assertFalse(summary["complete"])

    def test_wrapper_has_only_one_build_for_each_source(self):
        script = (ROOT / "run.sh").read_text()
        self.assertEqual(script.count('&& bash build.sh --use_sdk)'), 1)
        self.assertEqual(script.count('&& bash build.sh --use_mpi)'), 1)
        self.assertNotIn("for iteration", script)


if __name__ == "__main__":
    unittest.main()
