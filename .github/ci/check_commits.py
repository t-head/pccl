# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.

import os
import re
import subprocess
import sys


def git(*args):
    return subprocess.check_output(['git', *args], text=True)


commits = git('rev-list', os.environ['BASE_SHA'] + '..' + os.environ['HEAD_SHA']).splitlines()
errors = []
if len(commits) != 1:
    errors.append(
        f'PR must contain exactly one commit; found {len(commits)}. '
        'Squash the contribution commits before pushing.')
for sha in commits:
    message = git('show', '-s', '--format=%B', sha)
    lines = message.splitlines()
    if not lines or not lines[0].strip():
        errors.append(f'{sha[:12]}: title must not be empty')
    elif len(lines[0]) > 50:
        print(f'::notice::{sha[:12]}: title exceeds the recommended 50 characters; consider shortening it')
    if len(lines) > 1 and lines[1] != '':
        errors.append(f'{sha[:12]}: second line must be blank')
    issue_ref = (
        r'(?i:\b(?:Refs|Closes|Fixes))[ \t]+(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#[1-9][0-9]*\b'
        r'|https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/[1-9][0-9]*\b')
    if not re.search(issue_ref, '\n'.join(lines[1:])):
        errors.append(f'{sha[:12]}: missing issue reference in commit body')

    trailers = subprocess.check_output(
        ['git', 'interpret-trailers', '--parse', '--no-divider'], input=message, text=True).splitlines()
    signoffs = [match.groups() for line in trailers
                if (match := re.fullmatch(
                    r'Signed-off-by: (\S(?:.*\S)?) <([^<>\s]+@[^<>\s]+)>', line))]
    author = tuple(git('show', '-s', '--format=%an%x00%ae', sha).rstrip('\n').split('\0'))
    if not signoffs:
        errors.append(f'{sha[:12]}: missing valid Signed-off-by trailer')
    elif author not in signoffs:
        errors.append(f'{sha[:12]}: Signed-off-by name and email must match the commit author')
for error in errors:
    print(error)
print(f'Checked {len(commits)} contribution commits for message format and DCO.')
sys.exit(bool(errors))
