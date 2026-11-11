# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
from pathlib import Path
import re
import unittest

import run_cases


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'tests/xtest/daily-minimal-csv'
WORKFLOW = ROOT / '.github/workflows/m890p-daily-smoke.yml'
MPI_PROBE_WORKFLOW = ROOT / '.github/workflows/m890p-mpi-launch-probe.yml'


class DailySmokeTests(unittest.TestCase):
    def test_seven_supported_single_node_suites_are_selected(self):
        selected = run_cases.load_inputs(INPUT, 'standard', nodes=1)
        self.assertEqual(len(selected['cases']), 7)
        self.assertEqual(len({(row['plan'], row['suite']) for row in selected['cases']}), 7)
        cluster = run_cases.load_inputs(INPUT, 'standard', nodes=2)
        self.assertEqual(len(cluster['cases']), 1)
        self.assertEqual(len({(row['plan'], row['suite']) for row in cluster['cases']}), 1)
        self.assertTrue(all(row['suite_spec']['nodes'] == 2 for row in cluster['cases']))
        rdma = next(row for row in cluster['cases']
                    if row['suite'] == 'op_perf_rdma_shm_multi_nodes2_suite')
        self.assertIn('-np 16', rdma['cmd'])
        self.assertIn('-npernode 8', rdma['cmd'])
        self.assertIn('--hostfile {hostfile}', rdma['cmd'])
        self.assertIn('NCCL_MNNVL_ENABLE=0', rdma['cmd'])
        self.assertIn('all_reduce_perf', rdma['cmd'])
        self.assertIn('-mca orte_keep_fqdn_hostnames 1', rdma['cmd'])
        self.assertIn('-mca oob_tcp_if_include {mpi_interface}', rdma['cmd'])
        self.assertIn('-mca btl_tcp_if_include {mpi_interface}', rdma['cmd'])
        self.assertIn('-x NCCL_SOCKET_IFNAME={mpi_interface}', rdma['cmd'])
        self.assertIn('-x NCCL_NET=IB', rdma['cmd'])
        self.assertNotIn('NCCL_IB_GID_INDEX', rdma['cmd'])
        direct_multi = [row for row in cluster['cases'] if row['cmd'].startswith('mpirun ')]
        self.assertEqual(len(direct_multi), 1)
        for row in direct_multi:
            self.assertIn('-np 16', row['cmd'])
            self.assertIn('-npernode 8', row['cmd'])
            self.assertIn('--hostfile {hostfile}', row['cmd'])
        sanity = next(row for row in selected['cases'] if row['suite'] == 'sanity_m890p_register_suite')
        self.assertIn('--register 1', sanity['cmd'])
        self.assertEqual(sum(row['check'] == 'performance' for row in selected['cases']), 1)
        self.assertEqual(sum(row['check'] == 'exit_code' for row in selected['cases']), 6)
        self.assertTrue(all(row['suite_spec']['nodes'] == 1 for row in selected['cases']))
        self.assertEqual({row['suite_spec']['timeout'] for row in selected['cases']}, {300, 3600, 7200})

    def test_daily_commands_use_only_documented_runtime_bindings(self):
        with (INPUT / 'standard.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            self.assertNotIn('bond1', row['cmd'])
            self.assertNotIn('$PPU_HOME', row['cmd'])
            self.assertNotIn('/home/image', row['cmd'])
            self.assertNotIn('unit_test.py', row['cmd'])
        self.assertEqual(sum('{python_env}' in row['cmd'] for row in rows), 1)
        self.assertEqual(sum(row['cmd'].startswith('python3 run_test.py') for row in rows), 2)
        single_node_rows = run_cases.load_inputs(INPUT, 'standard', nodes=1)['cases']
        self.assertEqual(sum(row['cmd'].startswith('mpirun ') for row in single_node_rows), 4)
        self.assertEqual(sum(row['cmd'].startswith('mpirun ') for row in rows), 5)

    def test_daily_metadata_and_empty_notsm_are_explicit(self):
        metadata = json.loads((INPUT / 'export.json').read_text())
        self.assertEqual(metadata['xtest_version'], '0.12.0')
        self.assertTrue(re.fullmatch(r'[0-9a-f]{40}', metadata['source_revision']))
        with (INPUT / 'notsm.csv').open(newline='') as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])
        with self.assertRaisesRegex(ValueError, 'no Cases'):
            run_cases.load_inputs(INPUT, 'notsm', nodes=1)

    def test_daily_workflow_has_matching_gate_and_cleanup(self):
        text = WORKFLOW.read_text()
        self.assertIn('branches: [ci/m890p-daily-csv]', text)
        self.assertIn('tests/xtest/daily-minimal-csv', text)
        self.assertIn("expected = {'standard': 7, 'cluster': 1}", text)
        self.assertIn('len(metrics) != 1', text)
        self.assertIn('artifact_cache.py cleanup', text)
        self.assertIn('NEEDS: ${{ toJSON(needs) }}', text)
        self.assertIn('retention-days: 3', text)
        self.assertIn('name: m890p-daily-build-standard', text)
        self.assertIn('name: m890p-daily-build-${{ matrix.profile }}', text)
        self.assertIn('Two-node master-only daily smoke', text)
        self.assertNotIn('host_ipc: true', text)
        self.assertIn('host_network: true', text)
        self.assertIn('dns_policy: ClusterFirstWithHostNet', text)
        self.assertIn('ulimit -l unlimited', text)
        two_node = text.split('  two-node:\n', 1)[1].split('  summary:\n', 1)[0]
        two_node_setup = two_node.split('setup_command: |', 1)[1].split('\n          command: |', 1)[0]
        self.assertIn('export HWLOC_COMPONENTS=-gl', two_node_setup)
        self.assertIn('extra_env: NCCL_NET=IB', text)
        self.assertIn('ibv_devinfo -v', text)
        self.assertIn("raise SystemExit('No non-zero RoCE GID is visible inside the worker')", text)
        self.assertIn('timeout -k 5s 30s cluster-ssh "$host"', text)
        self.assertIn('ip -4 -o addr show dev $MPI_INTERFACE" < /dev/null', text)
        self.assertIn('test "$checked_hosts" -eq 2', text)
        self.assertIn('export OMPI_MCA_oob_tcp_if_include="$MPI_INTERFACE"', text)
        self.assertIn('timeout -k 10s 60s mpirun', text)
        probe = text.split('timeout -k 10s 60s mpirun', 1)[1].split('set +x', 1)[0]
        self.assertIn('export OMPI_MCA_orte_keep_fqdn_hostnames=1', text)
        self.assertIn('--mca orte_keep_fqdn_hostnames 1', probe)
        self.assertIn('--mca oob_tcp_if_include "$MPI_INTERFACE"', probe)
        self.assertIn('--mca btl_tcp_if_include "$MPI_INTERFACE"', probe)
        self.assertNotIn('human-review', text)
        self.assertNotIn('build-notsm', text)
        self.assertNotIn("expected = {('standard', 1): 71", text)

    def test_mpi_probe_does_not_run_pccl_or_trigger_the_full_daily(self):
        text = MPI_PROBE_WORKFLOW.read_text()
        self.assertIn('Two-node hostname launch without PCCL', text)
        self.assertIn('host_network: true', text)
        setup = text.split('setup_command: |', 1)[1].split('\n          command: |', 1)[0]
        self.assertIn('export HWLOC_COMPONENTS=-gl', setup)
        self.assertIn('cluster-ssh "$host"', text)
        self.assertIn('< /dev/null || exit 1', text)
        self.assertIn('getent ahosts "$short"', text)
        self.assertIn('run_probe local-no-gl', text)
        self.assertIn('cat > /tmp/cluster-ssh-trace', setup)
        self.assertNotIn('cat > /tmp/cluster-ssh-trace', text.split('\n          command: |', 1)[1])
        dual = text.split('run_probe dual-fqdn ', 1)[1]
        self.assertIn('timeout -k 2s 40s', dual)
        self.assertIn('--mca orte_keep_fqdn_hostnames 1', dual)
        self.assertIn('--mca oob_tcp_if_include "$MPI_INTERFACE"', dual)
        self.assertIn('--mca btl_tcp_if_include "$MPI_INTERFACE"', dual)
        self.assertIn('|| overall=1', dual)
        self.assertNotIn('run_cases.py', text)
        self.assertNotIn('build-standard', text)


if __name__ == '__main__':
    unittest.main()
