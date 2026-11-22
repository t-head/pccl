"""Generate and publish a squash commit body for a multi-commit PR."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from copilot_cli import final_response
from maintainer_approval import CHECK, event_pr_number

COMMENT_MARKER = '<!-- pccl-squash-message -->'
SKILL = Path(__file__).resolve().parents[2] / '.agents/skills/pccl-squash-message/SKILL.md'


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


def snapshot(api, number, *, automatic=False):
    if type(number) is not int or number <= 0:
        raise ValueError('PR number must be a positive integer')
    pr = api.request(f'/repos/{api.repository}/pulls/{number}')
    if pr['base']['repo']['full_name'] != api.repository:
        raise ValueError('Expected a PR in this repository')
    if pr['state'] != 'open' or pr['draft']:
        if automatic:
            return None
        raise ValueError('Expected an open, non-draft PR in this repository')
    return {'repository': api.repository, 'number': number, 'head': pr['head']['sha'],
            'base': pr['base']['sha'], 'target': pr['base']['ref'], 'file_count': pr['changed_files'],
            'commit_count': pr['commits']}


# Read-only AI generation

def copilot_response(output):
    message = final_response(output, allow_tools=True).strip()
    if message.startswith('ERROR:'):
        raise ValueError('Copilot could not prepare a complete message body: ' + message[:500])
    if sum(bool(line.strip()) for line in message.split('\n')) < 2:
        raise ValueError('Expected a multiline squash commit body')
    return message + '\n'


def ask_copilot(data):
    """Run the skill once, with only read-only GitHub tools available."""
    subprocess.run(['bash', str(Path(__file__).with_name('install_copilot.sh'))], check=True, timeout=300)
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


def review_gate_status(api, data):
    pages = api.request(f"/repos/{api.repository}/commits/{data['head']}/status?per_page=100", paginate=True)
    return next((status for page in pages for status in page['statuses'] if status['context'] == CHECK), {})


# Conversation preview and old-preview cleanup

def preview_comments(api, number):
    pages = api.request(f'/repos/{api.repository}/issues/{number}/comments?per_page=100', paginate=True)
    return [item for page in pages for item in page
            if item['user']['login'] == 'github-actions[bot]' and item['user']['type'] == 'Bot'
            and item.get('body', '').replace('\r\n', '\n').startswith(COMMENT_MARKER + '\n')]


def comment_body(data, message):
    longest = max((len(run) for run in re.findall(r'`+', message)), default=0)
    fence = '`' * max(3, longest + 1)
    return (f"{COMMENT_MARKER}\n"
            '### Suggested squash commit body\n\n'
            f"For commit `{data['head'][:12]}`.\n\n"
            "Paste this body into the description field in GitHub's Squash and merge dialog. "
            "Review the GitHub-filled title separately.\n\n"
            f"{fence}text\n{message.rstrip()}\n{fence}\n")


def refresh(api, number, ask=ask_copilot, *, automatic=False, cleanup_only=False):
    data = snapshot(api, number, automatic=automatic)
    if data is None or data['target'] != 'main':
        return None
    status = review_gate_status(api, data)
    if cleanup_only and status.get('target_url') != (
            f"https://github.com/{api.repository}/actions/runs/{os.environ['GITHUB_RUN_ID']}"):
        return None
    gate_state = status.get('state')
    if gate_state in {'error', 'failure'}:
        for comment in preview_comments(api, number):
            api.request(f"/repos/{api.repository}/issues/comments/{comment['id']}", 'DELETE')
        return None
    if cleanup_only or data['commit_count'] <= 1 or gate_state != 'success':
        return None
    comments = preview_comments(api, number)
    message = ask(data)
    body = comment_body(data, message)
    if len(body.encode('utf-8')) > 65535:
        raise ValueError('Squash body exceeds the comment size limit')
    published = api.request(f'/repos/{api.repository}/issues/{number}/comments',
                            'POST', {'body': body})
    # Preserve old previews if publishing their replacement fails.
    for comment in comments:
        api.request(f"/repos/{api.repository}/issues/comments/{comment['id']}", 'DELETE')
    return published['html_url']


# Command-line entry point

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pr', type=int, default=os.environ.get('PR_NUMBER') or None)
    parser.add_argument('--cleanup-only', action='store_true')
    parser.add_argument('--repository', default=os.environ.get('GITHUB_REPOSITORY', 't-head/pccl'))
    args = parser.parse_args()
    api = GitHub(args.repository, os.environ.get('GITHUB_TOKEN', ''))
    number = args.pr
    if number is None:
        run = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())['workflow_run']
        number = event_pr_number(run, api.repository,
                                 {'.github/workflows/review-gate.yml': {'workflow_run', 'workflow_dispatch'}})
    url = refresh(api, number, automatic=args.cleanup_only or os.environ.get('GITHUB_EVENT_NAME') == 'workflow_run',
                  cleanup_only=args.cleanup_only)
    print('Published squash message: ' + url if url else 'No squash preview published.')


if __name__ == '__main__':
    main()
