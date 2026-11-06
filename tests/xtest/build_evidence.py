# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind a restored build archive to its source revisions and build profile."""

import argparse
import hashlib
import json
from pathlib import Path
import re

from manifest import read_json


FIELDS = {'schema_version', 'pccl_revision', 'pccl_tests_revision', 'build_profile', 'artifact_sha256'}
HEX40 = re.compile(r'[0-9a-f]{40}\Z')
HEX64 = re.compile(r'[0-9a-f]{64}\Z')


def file_sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def validate(record, inputs, profile, archive=None):
    expected_revisions = {name: inputs.get(name) for name in ('pccl_revision', 'pccl_tests_revision')}
    if (not isinstance(record, dict) or set(record) != FIELDS
            or type(record['schema_version']) is not int or record['schema_version'] != 1
            or record['build_profile'] != profile
            or any(not isinstance(record[name], str) or not HEX40.fullmatch(record[name])
                   or record[name] != expected_revisions[name]
                   for name in ('pccl_revision', 'pccl_tests_revision'))
            or not isinstance(record['artifact_sha256'], str)
            or not HEX64.fullmatch(record['artifact_sha256'])):
        raise ValueError('Build evidence differs from the manifest revision or profile')
    if archive is not None:
        path = Path(archive)
        if not path.is_file() or file_sha256(path) != record['artifact_sha256']:
            raise ValueError('Build archive differs from its recorded SHA-256')
    return dict(record)


def create(archive, pccl_revision, pccl_tests_revision, profile):
    record = dict(schema_version=1, pccl_revision=pccl_revision,
                  pccl_tests_revision=pccl_tests_revision, build_profile=profile,
                  artifact_sha256=file_sha256(archive))
    inputs = {'pccl_revision': pccl_revision, 'pccl_tests_revision': pccl_tests_revision}
    return validate(record, inputs, profile, archive)


def load(path, inputs, profile, archive):
    return validate(read_json(path), inputs, profile, archive)


def write_json(path, record):
    path = Path(path)
    if not path.parent.is_dir():
        raise ValueError('Build evidence output requires an existing directory')
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('create',))
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--pccl-revision', required=True)
    parser.add_argument('--pccl-tests-revision', required=True)
    parser.add_argument('--profile', choices=('standard', 'notsm'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = create(args.archive, args.pccl_revision, args.pccl_tests_revision, args.profile)
    write_json(args.output, record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
