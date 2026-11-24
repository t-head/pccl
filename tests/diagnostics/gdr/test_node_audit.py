import ast
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from node_audit import FAILED_HOSTS, PAIR, PROBE, classify, has_gpu_request, kube, pod_manifest


class NodeAuditTests(unittest.TestCase):
    def test_module_detection_requires_a_real_host_view(self):
        self.assertEqual(classify({"alixpu_peermem", "mlx5_core"}, False), "loaded")
        self.assertEqual(classify(None, True), "loaded")
        self.assertEqual(classify({"mlx5_core", "ib_core"}, False), "not_loaded")
        self.assertEqual(classify(set(), False), "unknown")
        self.assertEqual(classify(None, False), "unknown")

    def test_probes_are_cpu_only_nonprivileged_and_explicitly_pinned(self):
        pod = pod_manifest("audit-test", "ppu-sched", "123", "node-03", "sdk-image")
        self.assertFalse(has_gpu_request(pod))
        self.assertFalse(pod["spec"]["automountServiceAccountToken"])
        self.assertEqual(pod["spec"]["nodeSelector"], {"kubernetes.io/hostname": "node-03"})
        security = pod["spec"]["containers"][0]["securityContext"]
        self.assertTrue(security["readOnlyRootFilesystem"])
        self.assertFalse(security["allowPrivilegeEscalation"])
        self.assertEqual(security["capabilities"]["drop"], ["ALL"])
        self.assertNotIn("volumes", pod["spec"])
        self.assertNotIn("hostPID", pod["spec"])

    def test_admission_injected_gpu_is_detected(self):
        pod = pod_manifest("audit-test", "ppu-sched", "123", "node-03", "sdk-image")
        pod["spec"]["containers"][0]["resources"]["limits"]["alibabacloud.com/ppu"] = "1"
        self.assertTrue(has_gpu_request(pod))

    def test_requested_hosts_and_probe_syntax(self):
        self.assertEqual(len(FAILED_HOSTS), 16)
        self.assertEqual(len(set(FAILED_HOSTS)), 16)
        self.assertTrue(set(PAIR) <= set(FAILED_HOSTS))
        ast.parse(PROBE)
        self.assertNotIn("modprobe", PROBE)
        self.assertNotIn("rmmod", PROBE)

    def test_cluster_credentials_are_not_broken_by_timeout_flags(self):
        with patch("node_audit.subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout="{}", stderr="")) as command:
            self.assertEqual(kube("get", "nodes", "-o", "json"), "{}")
        self.assertEqual(command.call_args.args[0], ["kubectl", "get", "nodes", "-o", "json"])
        self.assertEqual(command.call_args.kwargs["timeout"], 35)


if __name__ == "__main__":
    unittest.main()
