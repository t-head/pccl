#!/usr/bin/env python3
"""Bounded, same-allocation comparison of verified historical PCCL archives."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import artifact_cache
import run_cases

ARCHIVES = {
    'good': ('abb6688b4b5419664e7bf13c2d9107bd7acafd89',
             '20b03ec2461a4d827037da4ba6bf2311c31f7c87852b1c5491255870bd93a548'),
    'failed': ('30ac5589671c608c51266cfd99f3b0b59eddc981',
               'f083b7721f306ce59362060165f332bd5dc758e1e79a83a2433578a9a398fffa'),
}


def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def cases():
    mpi = ('mpirun --allow-run-as-root -np 16 -npernode 8 --hostfile {hostfile} '
           '-mca btl_tcp_if_include {mpi_interface} -mca btl ^openib '
           '-x PATH={path} -x LD_LIBRARY_PATH={ld_library_path} '
           '-x PCCL_GDR_CPU_FLUSH=1 -x PCCL_CPU_FLUSH_ASYNC=1 '
           '-x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_OOO_RQ=1 '
           '-x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,NET,REG ')
    sweep = 'all_gather_perf -b 8 -e 16M -f 2 -t 1 -g 1 -n 3 -w 1'
    return [
        ('gdr_small', mpi + '-x NCCL_NET_GDR_LEVEL=4 ' + sweep),
        ('host_small', mpi + '-x NCCL_NET_GDR_LEVEL=0 ' + sweep),
        ('gdr_no_relaxed_ordering', mpi + '-x NCCL_NET_GDR_LEVEL=4 '
         '-x NCCL_IB_PCI_RELAXED_ORDERING=0 ' + sweep),
        ('gdr_256mb', mpi + '-x NCCL_NET_GDR_LEVEL=4 '
         'all_gather_perf -b 256M -e 256M -t 1 -g 1 -n 3 -w 1'),
        ('gin3', mpi + '-x PCCL_GIN_TYPE=3 '
         'alltoall_perf -b 8 -e 16M -f 2 --elapsed_type 1 -R 2 -t 1 -g 1 -D 3 -V 1 -n 3 -w 1'),
    ]


def run(args):
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sequence = []
    for label in ('good-first', 'failed', 'good-repeat'):
        archive_name = 'good' if label.startswith('good') else 'failed'
        revision, digest = ARCHIVES[archive_name]
        cache = args.cache / archive_name
        archive = artifact_cache.validate(cache, 'standard', revision,
                                          artifact_cache.PCCL_TESTS_REVISION, digest)
        extraction = root / 'shared' / label
        extraction.mkdir(parents=True)
        subprocess.run(['tar', '-xzf', str(archive), '-C', str(extraction)], check=True)
        build = extraction / 'output_standard'
        for directory, expected in ((build / 'pccl', revision),
                                    (build / 'pccl-tests', artifact_cache.PCCL_TESTS_REVISION)):
            actual = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != expected:
                raise ValueError(f'Extracted source mismatch: {actual} != {expected}')
        inputs = root / 'inputs' / label
        (inputs / 'envs').mkdir(parents=True)
        (inputs / 'envs' / 'comparison.env').write_text('UMD_PLATFORM_TYPE=1\n')
        (inputs / 'export.json').write_text(json.dumps({
            'format_version': 1, 'source_revision': revision,
            'runner_revision': artifact_cache.PCCL_TESTS_REVISION, 'xtest_version': 'not-used'}))
        write_csv(inputs / 'suites.csv', run_cases.SUITE_FIELDS, [{
            'plan': 'rdma_ab', 'suite': label, 'nodes': 2, 'cards_per_node': 8,
            'timeout': 90, 'env': 'comparison.env'}])
        rows = [{'plan': 'rdma_ab', 'suite': label, 'name': name,
                 'check': 'validation_summary', 'cmd': command} for name, command in cases()]
        write_csv(inputs / 'standard.csv', run_cases.CASE_FIELDS, rows)
        environment = dict(os.environ)
        environment['LD_LIBRARY_PATH'] = str(build / 'lib') + ':' + environment.get('LD_LIBRARY_PATH', '')
        selected = run_cases.load_inputs(inputs, 'standard')
        results = run_cases.run_all(selected, build, build / 'pccl-tests/build', root / 'results' / label,
                                   args.hostfile, environment, mpi_interface=args.interface,
                                   build_identity=artifact_cache.read_record(cache / 'output_standard.build.json'))
        safe = True
        for record in results:
            recovery = root / 'results' / label / 'rdma_ab' / label / record['name'] / 'recovery.json'
            if recovery.is_file() and not json.loads(recovery.read_text()).get('safe'):
                safe = False
        sequence.append({'label': label, 'revision': revision, 'archive_sha256': digest,
                         'safe': safe, 'results': results})
        if not safe or any(record['status'] in ('CANCELLED', 'NOT_RUN') for record in results):
            break
    summary = {'sequence': sequence,
               'complete': len(sequence) == 3 and all(
                   item['safe'] and len(item['results']) == 5
                   and not any(row['status'] in ('CANCELLED', 'NOT_RUN') for row in item['results'])
                   for item in sequence),
               'comparison_only': True, 'limits': {'nodes': 2, 'cards_per_node': 8,
                                                  'cases_per_build': 5, 'case_timeout_seconds': 90}}
    (root / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0 if summary['complete'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--hostfile', type=Path, required=True)
    parser.add_argument('--interface', required=True)
    raise SystemExit(run(parser.parse_args()))
