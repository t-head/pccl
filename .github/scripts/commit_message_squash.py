# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
"""Generate and publish a combined commit message for a multi-commit PR."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

COMMENT_MARKER = '<!-- pccl-squash-message -->'
SKILL = Path(__file__).resolve().parents[2] / '.agents/skills/pccl-squash-message/SKILL.md'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# GitHub access and input snapshot

class GitHub:
    def __init__(self, repository, token):
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', repository) or not token:
            raise ValueError('A GitHub repository and token are required')
        self.repository, self.token = repository, token

    def request(self, path, method='GET', payload=None, *, paginate=False):
        args = ['gh', 'api', path, '--method', method]
        if payload is not None:
            args += ['--input', '-']
        if paginate:
            args += ['--paginate', '--slurp']
        try:
            output = subprocess.run(
                args, input=json.dumps(payload) if payload is not None else None,
                text=True, capture_output=True, timeout=60, check=True,
                env={**os.environ, 'GH_TOKEN': self.token, 'GH_HOST': 'github.com'}).stdout
        except subprocess.CalledProcessError as error:
            detail = error.stderr.replace(self.token, '[redacted]').strip()[:500]
            raise ValueError('GitHub command failed: ' + detail) from None
        return json.loads(output) if output.strip() else None


def snapshot(api, number):
    """Bind the message to all PR inputs that can affect its content."""
    if type(number) is not int or number <= 0:
        raise ValueError('PR number must be a positive integer')
    pr = api.request(f'/repos/{api.repository}/pulls/{number}')
    if pr['state'] != 'open' or pr['draft'] or pr['base']['repo']['full_name'] != api.repository:
        raise ValueError('Expected an open, non-draft PR in this repository')
    version = [api.repository, number, pr['title'], pr['body'], pr['commits'],
               pr['changed_files'], pr['head']['sha'], pr['base']['sha'], pr['base']['ref']]
    return {'repository': api.repository, 'number': number, 'input_digest': digest(version),
            'head': pr['head']['sha'],
            'base': pr['base']['sha'], 'target': pr['base']['ref'], 'file_count': pr['changed_files'],
            'commit_count': pr['commits']}


# Read-only AI generation

def split_message(message):
    title, separator, body = message.rstrip('\n').partition('\n\n')
    if not separator or '\n' in title or not title.strip() or not body.strip():
        raise ValueError('Expected a title, blank line and body')
    return title, body


def copilot_response(output):
    events = [json.loads(line) for line in output.splitlines() if line.strip()]
    answers = [event['data']['content'] for event in events
               if event.get('type') == 'assistant.message' and not event['data'].get('toolRequests')
               and event['data'].get('content', '').strip()]
    message = (answers[-1] if answers else '').strip()
    if message.startswith('ERROR:'):
        raise ValueError('Copilot could not prepare a complete message: ' + message[:500])
    split_message(message)
    return message + '\n'


def ask_copilot(data):
    """Run the skill once, with only read-only GitHub tools available."""
    prompt = SKILL.read_text() + '\n\nExpected PR snapshot:\n' + json.dumps(data)
    with tempfile.TemporaryDirectory(prefix='pccl-squash-') as directory:
        result = subprocess.run([
            os.environ['COPILOT_CLI'], '--no-auto-update', '--prompt=' + prompt,
            '--no-ask-user', '--no-custom-instructions', '--disable-mcp-server', 'githubiq',
            '--add-github-mcp-tool', 'pull_request_read', '--add-github-mcp-tool', 'get_file_contents',
            '--available-tools=github-mcp-server-pull_request_read,github-mcp-server-get_file_contents',
            '--allow-tool=github-mcp-server(pull_request_read),github-mcp-server(get_file_contents)',
            '--no-color', '--stream', 'off', '--output-format', 'json',
            '--log-level', 'error', '--log-dir', directory], cwd=directory,
            capture_output=True, text=True, timeout=300,
            env={'PATH': os.environ.get('PATH', ''), 'GITHUB_TOKEN': os.environ['GITHUB_TOKEN'],
                 'COPILOT_LARGE_OUTPUT_THRESHOLD_BYTES': '120000',
                 'COPILOT_HOME': directory, 'COPILOT_AUTO_UPDATE': 'false', 'CI': 'true', 'NO_COLOR': '1'})
        if result.returncode:
            detail = ' '.join(result.stderr.replace(os.environ['GITHUB_TOKEN'], '[redacted]').split())[:500]
            raise ValueError(f'Copilot failed (exit {result.returncode}): {detail or "No error detail returned"}')
        return copilot_response(result.stdout)


# Prepare a draft; publishing checks that its input version is still current.

def generate(api, number, ask=ask_copilot):
    data = snapshot(api, number)
    # One commit already has its final message; only remove any old preview.
    message = ask(data) if data['commit_count'] > 1 else None
    return {**data, 'message': message}


# Conversation preview and old-preview cleanup

def preview_comments(api, number):
    pages = api.request(f'/repos/{api.repository}/issues/{number}/comments?per_page=100', paginate=True)
    return [item for page in pages for item in page
            if item['user']['login'] == 'github-actions[bot]' and item['user']['type'] == 'Bot'
            and item.get('body', '').startswith(COMMENT_MARKER + '\n')]


def comment_body(draft):
    return (f"{COMMENT_MARKER}\n<!-- version: {draft['input_digest']} -->\n"
            '### Suggested squash commit message\n\n'
            f"For commit `{draft['head'][:12]}`.\n\n"
            f"<pre>{html.escape(draft['message'])}</pre>\n\n"
            'Review this message, then comment `/merge` to request squash merging. '
            'Available after the merge workflow is on `main`; '
            'the web Squash and merge button remains available.\n')


def publish(api, draft, expected_pr):
    comments = preview_comments(api, expected_pr)
    data = snapshot(api, expected_pr)
    if any(draft.get(key) != value for key, value in data.items()):
        raise ValueError('PR inputs changed; generate a fresh draft')
    published = None
    if data['commit_count'] > 1:
        # Always publish at the conversation end before removing old previews.
        published = api.request(f'/repos/{api.repository}/issues/{expected_pr}/comments',
                                'POST', {'body': comment_body(draft)})
    for comment in comments:
        api.request(f"/repos/{api.repository}/issues/comments/{comment['id']}", 'DELETE')
    return published['html_url'] if published else None


# Command-line entry point

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pr', type=int)
    mode.add_argument('--publish', action='store_true')
    parser.add_argument('--expected-pr', type=int)
    parser.add_argument('--repository', default=os.environ.get('GITHUB_REPOSITORY', 't-head/pccl'))
    args = parser.parse_args()
    api = GitHub(args.repository, os.environ.get('GITHUB_TOKEN', ''))
    if args.publish:
        url = publish(api, json.loads(os.environ['SQUASH_DRAFT']), args.expected_pr)
        print('Published squash message: ' + url if url else
              'Single-commit PR: no AI draft comment; any previous preview was removed.')
    else:
        print(json.dumps(generate(api, args.pr), ensure_ascii=True))


if __name__ == '__main__':
    main()
