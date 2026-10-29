# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
"""Merge the displayed squash message after an authorized /merge comment."""

from datetime import datetime
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

import prepare_squash as squash


def github_cli(api, *args):
    return subprocess.run(
        ['gh', *args], env={**os.environ, 'GH_TOKEN': api.token, 'GH_HOST': 'github.com'},
        capture_output=True, timeout=60, check=True).stdout


def command(api, event):
    expected = event['comment']
    number = event['issue']['number']
    if (event['action'] != 'created' or not event['issue'].get('pull_request') or
            event['repository']['full_name'] != api.repository or
            type(number) is not int or number <= 0 or
            type(expected['id']) is not int or expected['id'] <= 0):
        raise ValueError('Expected a new /merge comment on a pull request')
    comment = api.get(f"/repos/{api.repository}/issues/comments/{expected['id']}")
    if (comment['user']['type'] != 'User' or comment['user']['id'] != expected['user']['id'] or
            comment['body'] != '/merge' or comment['body'] != expected['body'] or
            comment['created_at'] != comment['updated_at'] or
            comment['created_at'] != expected['created_at'] or
            comment['issue_url'] != f'https://api.github.com/repos/{api.repository}/issues/{number}'):
        raise ValueError('The command changed or is not a new human merge request')
    login = comment['user']['login']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}', login):
        raise ValueError('Invalid command author')
    permission = api.get(f'/repos/{api.repository}/collaborators/{login}/permission')
    if permission['permission'] not in {'write', 'admin'}:
        raise ValueError('The command author needs repository Write, Maintain or Admin access')
    return comment


def preview(api, number, request):
    comments = [item for item in api.pages(f'/repos/{api.repository}/issues/{number}/comments')
                if item['user']['login'] == 'github-actions[bot]' and item['user']['type'] == 'Bot'
                and item.get('body', '').startswith(squash.COMMENT_MARKER + '\n')]
    if len(comments) != 1:
        raise ValueError('One displayed AI draft is required; run Prepare squash commit first')
    draft_comment = comments[0]
    # Equal timestamps are ambiguous at GitHub's second-level precision.
    if (datetime.fromisoformat(draft_comment['updated_at'].replace('Z', '+00:00')) >=
            datetime.fromisoformat(request['created_at'].replace('Z', '+00:00'))):
        raise ValueError('The draft changed after this command; review it and send /merge again')
    return draft_comment


def load_draft(api, number, draft_comment):
    match = re.search(r'^<!-- run: ([1-9][0-9]*) -->$', draft_comment['body'], re.M)
    if not match:
        raise ValueError('The displayed draft has no generation record')
    run_id = match[1]
    run = api.get(f'/repos/{api.repository}/actions/runs/{run_id}')
    if (run['path'] != '.github/workflows/prepare-squash.yml' or
            run['repository']['full_name'] != api.repository or
            run['status'] != 'completed' or run['conclusion'] != 'success' or
            run['event'] not in {'pull_request', 'workflow_dispatch'} or
            (run['event'] == 'pull_request' and run['head_repository']['full_name'] != api.repository) or
            (run['event'] == 'workflow_dispatch' and run['head_branch'] != 'main')):
        raise ValueError('The draft must come from a successful message preparation workflow')
    artifacts = api.get(f'/repos/{api.repository}/actions/runs/{run_id}/artifacts?per_page=100')
    candidates = [item for item in artifacts['artifacts']
                  if item['name'] == 'squash-message' and not item['expired']]
    if artifacts['total_count'] > 100 or len(candidates) != 1:
        raise ValueError('The squash draft artifact is missing or expired; generate it again')
    artifact = candidates[0]
    if not 0 < artifact['size_in_bytes'] <= 262144:
        raise ValueError('The squash draft artifact is too large')
    archive = github_cli(api, 'api', f"/repos/{api.repository}/actions/artifacts/{artifact['id']}/zip")
    if len(archive) > 262144:
        raise ValueError('The squash draft download is too large')
    with zipfile.ZipFile(io.BytesIO(archive)) as files:
        entries = [entry for entry in files.infolist() if entry.filename == 'proposal.json']
        if len(entries) != 1 or not 0 < entries[0].file_size <= 65536:
            raise ValueError('Expected one bounded proposal.json in the draft artifact')
        draft = json.loads(files.read(entries[0]))
    if (draft.get('number') != number or draft.get('repository') != api.repository or
            draft.get('commit_count', 0) < 2 or
            draft_comment['body'] != squash.comment_body(draft, run_id)):
        raise ValueError('The displayed message does not match this PR draft artifact')
    return draft


def check_readiness(api, number, draft):
    current = json.loads(github_cli(
        api, 'pr', 'view', str(number), '--repo', api.repository, '--json',
        'state,isDraft,headRefOid,baseRefOid,baseRefName,reviewDecision,mergeStateStatus'))
    if current['state'] != 'OPEN' or current['isDraft'] or current['baseRefName'] != 'main':
        raise ValueError('Only an open, non-draft PR targeting main can be merged')
    if current['headRefOid'] != draft['head'] or current['baseRefOid'] != draft['base']:
        raise ValueError('The PR changed; generate a fresh draft and send /merge again')
    if current['reviewDecision'] != 'APPROVED' or current['mergeStateStatus'] != 'CLEAN':
        raise ValueError('PR is not ready: reviews=' + str(current['reviewDecision']) +
                         ', merge status=' + str(current['mergeStateStatus']) +
                         '. Complete the required reviews and checks, then send /merge again')


def merge_message(api, number, draft):
    title, body = draft['message'].rstrip('\n').split('\n\n', 1)
    request = Request(f'https://api.github.com/repos/{api.repository}/pulls/{number}/merge',
                      data=json.dumps({'sha': draft['head'], 'merge_method': 'squash',
                                       'commit_title': title, 'commit_message': body}).encode(),
                      headers={'Authorization': 'Bearer ' + api.token,
                               'Accept': 'application/vnd.github+json',
                               'Content-Type': 'application/json',
                               'X-GitHub-Api-Version': '2022-11-28'}, method='PUT')
    try:
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
    except HTTPError as error:
        raise ValueError(f'GitHub refused the squash merge (HTTP {error.code}); check the PR') from None
    if result.get('merged') is not True:
        raise ValueError('GitHub did not merge the PR; check its current state')
    return result['sha']


def merge_request(api, event):
    request = command(api, event)
    number = event['issue']['number']
    draft_comment = preview(api, number, request)
    draft = load_draft(api, number, draft_comment)
    squash.verify(api, draft)
    check_readiness(api, number, draft)
    # Recheck mutable input before writing; GitHub atomically checks HEAD and branch rules.
    command(api, event)
    current_comment = preview(api, number, request)
    if any(current_comment[key] != draft_comment[key] for key in ('id', 'body', 'updated_at')):
        raise ValueError('The displayed draft changed; review it and send /merge again')
    squash.verify(api, draft)
    check_readiness(api, number, draft)
    return merge_message(api, number, draft)


def main():
    if os.environ.get('GITHUB_EVENT_NAME') != 'issue_comment':
        raise ValueError('Only a new PR command comment can request merging')
    api = squash.GitHub(os.environ['GITHUB_REPOSITORY'], os.environ['GITHUB_TOKEN'])
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    try:
        sha = merge_request(api, event)
        result = f"Squash-merged PR #{event['issue']['number']} using the confirmed message: `{sha}`."
        status = 0
    except Exception as error:
        result = 'Could not merge: ' + (str(error) if isinstance(error, ValueError)
                                      else 'Verification failed; see the merge workflow logs')
        status = 1
    print(result)
    try:
        api.write_comment(f"/repos/{api.repository}/issues/{event['issue']['number']}/comments", result)
    except Exception:
        print('Could not publish the result comment; check the PR and workflow logs.', file=sys.stderr)
    return status


if __name__ == '__main__':
    sys.exit(main())
