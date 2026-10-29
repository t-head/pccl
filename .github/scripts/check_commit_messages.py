# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
"""Share message-format checks; validate original commit DCO in PR CI."""

import os
import re
import subprocess
import sys

ISSUE_URL = r'(?i:https://github\.com)/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/[1-9][0-9]*\b'
ISSUE_REFERENCE = re.compile(
    r'(?i:\b(?:Refs|Closes|Fixes))[ \t]+(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#[1-9][0-9]*\b'
    r'|' + ISSUE_URL)
IDENTITY = r'(\S(?:.*\S)?) <([^<>\s]+@[^<>\s]+)>'


def git(*args, message=None):
    return subprocess.check_output(['git', *args], input=message, text=True)


def trailers(message):
    return git('interpret-trailers', '--parse', '--no-divider', message=message).splitlines()


def validate_message(message):
    lines = message.splitlines()
    errors = []
    if not lines or not lines[0].strip():
        errors.append('title must not be empty')
    if len(lines) > 1 and lines[1] != '':
        errors.append('second line must be blank')
    if not ISSUE_REFERENCE.search('\n'.join(lines[1:])):
        errors.append('missing issue reference in commit body')
    return errors


def validate_dco(message, author):
    """Check source-commit sign-offs in the dedicated PR check."""
    signoffs = [match.groups() for line in trailers(message)
                if (match := re.fullmatch('Signed-off-by: ' + IDENTITY, line))]
    if not signoffs:
        return ['missing valid Signed-off-by trailer']
    if tuple(author) not in signoffs:
        return ['Signed-off-by name and email must match the commit author']
    return []


def check_range(base, head):
    # Resolve object IDs before passing the range to Git; there is no commit-count cap.
    base = git('rev-parse', '--verify', '--end-of-options', base + '^{commit}').strip()
    head = git('rev-parse', '--verify', '--end-of-options', head + '^{commit}').strip()
    commits = git('rev-list', base + '..' + head).splitlines()
    errors = [] if commits else ['PR must contain at least one contribution commit.']
    for sha in commits:
        message = git('show', '-s', '--format=%B', sha)
        author = git('show', '-s', '--format=%an%x00%ae', sha).rstrip('\n').split('\0')
        errors.extend(f'{sha[:12]}: {error}' for error in
                      validate_message(message) + validate_dco(message, author))
        if message.splitlines() and len(message.splitlines()[0]) > 50:
            print(f'::notice::{sha[:12]}: title exceeds the recommended 50 characters; consider shortening it')
    for error in errors:
        print(error)
    print(f'Checked {len(commits)} contribution commits for message format and DCO.')
    return bool(errors)


if __name__ == '__main__':
    sys.exit(check_range(os.environ['BASE_SHA'], os.environ['HEAD_SHA']))
