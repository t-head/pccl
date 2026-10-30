# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
"""Draft a squash message and publish its preview to the PR."""

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
    version = [pr[key] for key in ('title', 'body', 'commits', 'changed_files')] + [
        pr['head']['sha'], pr['base']['sha'], pr['base']['ref']]
    return {'repository': api.repository, 'number': number, 'input_digest': digest(version),
            'head': pr['head']['sha'],
            'base': pr['base']['sha'], 'target': pr['base']['ref'], 'file_count': pr['changed_files'],
            'commit_count': pr['commits']}


# Read-only AI generation

def copilot_response(output):
    events = [json.loads(line) for line in output.splitlines() if line.strip()]
    answers = [event['data']['content'] for event in events
               if event.get('type') == 'assistant.message' and not event['data'].get('toolRequests')
               and event['data'].get('content', '').strip()]
    message = (answers[-1] if answers else '').strip()
    if message.startswith('ERROR:'):
        raise ValueError('Copilot could not prepare a complete message: ' + message[:500])
    title, separator, body = message.partition('\n\n')
    if not separator or '\n' in title or not title.strip() or not body.strip():
        raise ValueError('Copilot must return a title, blank line and body')
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


# Prepare a draft and recheck it before publishing or merging

def generate(api, number, ask=ask_copilot):
    data = snapshot(api, number)
    # One commit already has its final message; only remove any old preview.
    message = ask(data) if data['commit_count'] > 1 else None
    # The PR may change while the model is running.
    if data != snapshot(api, number):
        raise ValueError('PR inputs changed during generation; run again')
    return {**data, 'message': message, 'message_digest': digest(message)}


def verify(api, draft):
    """Reject stale or altered drafts before any comment or merge write."""
    if draft['repository'] != api.repository:
        raise ValueError('Draft repository does not match')
    data = snapshot(api, draft['number'])
    if any(draft.get(key) != value for key, value in data.items()):
        raise ValueError('PR inputs changed; generate a fresh draft')
    if data['commit_count'] == 1:
        if draft['message'] is not None:
            raise ValueError('A single-commit PR must not publish an AI draft')
    elif not isinstance(draft['message'], str) or not draft['message'].strip():
        raise ValueError('A multi-commit PR requires a message')
    # This detects accidental edits, not authenticated AI output.
    if draft.get('message_digest') != digest(draft['message']):
        raise ValueError('The displayed message changed; generate a fresh draft')
    return data['commit_count']


# Conversation preview and old-preview cleanup

def preview_comments(api, number):
    pages = api.request(f'/repos/{api.repository}/issues/{number}/comments?per_page=100', paginate=True)
    return [item for page in pages for item in page
            if item['user']['login'] == 'github-actions[bot]' and item['user']['type'] == 'Bot'
            and item.get('body', '').startswith(COMMENT_MARKER + '\n')]


def comment_body(draft, run_id):
    if not re.fullmatch(r'[1-9][0-9]*', str(run_id)):
        raise ValueError('Invalid workflow run ID')
    run_url = f"https://github.com/{draft['repository']}/actions/runs/{run_id}"
    # Keep verification data with the displayed message. Escape HTML delimiters
    # so model text cannot end this hidden comment or alter the visible preview.
    context = json.dumps({key: value for key, value in draft.items() if key != 'message'},
                         ensure_ascii=True, separators=(',', ':'))
    context = context.replace('<', '\\u003c').replace('>', '\\u003e')
    return (f'{COMMENT_MARKER}\n<!-- run: {run_id} -->\n<!-- draft: {context} -->\n'
            '### Suggested squash commit message\n\n'
            f"For commit `{draft['head'][:12]}`.\n\n"
            f"<pre>{html.escape(draft['message'])}</pre>\n\n"
            'Review this message, then comment `/merge` to request squash merging '
            'when required reviews and checks pass. Available after the merge workflow '
            'is on `main`; the web Squash and merge button remains available.\n\n'
            f'[View generation record]({run_url})\n')


def publish(api, draft, expected_pr, run_id):
    if type(expected_pr) is not int or expected_pr <= 0 or draft['number'] != expected_pr:
        raise ValueError('Draft does not match the workflow PR')
    if not re.fullmatch(r'[1-9][0-9]*', str(run_id)):
        raise ValueError('Invalid workflow run ID')
    comments = preview_comments(api, expected_pr)
    for comment in comments:
        previous_run = re.search(r'<!-- run: ([1-9][0-9]*) -->', comment['body'])
        if previous_run and int(previous_run[1]) > int(run_id):
            raise ValueError('A newer workflow has already published its draft')
    # Re-read the complete PR after locating the comment and before any write.
    current = None
    if verify(api, draft) > 1:
        body = comment_body(draft, run_id)
        current = next((comment for comment in comments if comment['body'] == body), None)
        # Publish at the conversation end before removing old previews. Reuse
        # an identical comment when retrying an interrupted cleanup.
        current = current or api.request(f'/repos/{api.repository}/issues/{expected_pr}/comments',
                                         'POST', {'body': body})
    for comment in comments:
        if current is None or comment['id'] != current['id']:
            api.request(f"/repos/{api.repository}/issues/comments/{comment['id']}", 'DELETE')
    return current['html_url'] if current else None


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
        url = publish(api, json.loads(os.environ['SQUASH_DRAFT']), args.expected_pr,
                      os.environ.get('GITHUB_RUN_ID', ''))
        print('Published squash message: ' + url if url else
              'Single-commit PR: no AI draft comment; any previous preview was removed.')
    else:
        print(json.dumps(generate(api, args.pr), ensure_ascii=True))


if __name__ == '__main__':
    main()
