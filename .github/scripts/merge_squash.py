# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
"""Merge the displayed squash message after an authorized /merge comment."""

from datetime import datetime
import html
import json
import os
from pathlib import Path
import re
import sys

import prepare_squash as squash


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
    body = draft_comment['body']
    run_match = re.search(r'^<!-- run: ([1-9][0-9]*) -->$', body, re.M)
    context = re.search(r'^<!-- draft: (\{[^\n]*\}) -->$', body, re.M)
    message = re.search(r'<pre>(.*?)</pre>', body, re.S)
    if not all((run_match, context, message)):
        raise ValueError('The preview has no complete draft record; generate it again')
    draft = json.loads(context[1])
    draft['message'] = html.unescape(message[1])
    run_id = run_match[1]
    if (draft.get('number') != number or draft.get('repository') != api.repository or
            draft.get('commit_count', 0) < 2 or body != squash.comment_body(draft, run_id)):
        raise ValueError('The preview does not match this PR draft')
    run = api.get(f'/repos/{api.repository}/actions/runs/{run_id}')
    if (run['path'] != '.github/workflows/prepare-squash.yml' or
            run['repository']['full_name'] != api.repository or
            run['status'] != 'completed' or run['conclusion'] != 'success' or
            run['event'] not in {'pull_request', 'workflow_dispatch'} or
            (run['event'] == 'pull_request' and run['head_repository']['full_name'] != api.repository) or
            (run['event'] == 'workflow_dispatch' and run['head_branch'] != 'main')):
        raise ValueError('The draft must come from a successful message preparation workflow')
    return draft


def check_readiness(api, number, draft):
    current = json.loads(api.cli(
        'pr', 'view', str(number), '--repo', api.repository, '--json',
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
    result = api.request(f'/repos/{api.repository}/pulls/{number}/merge', 'PUT',
                         {'sha': draft['head'], 'merge_method': 'squash',
                          'commit_title': title, 'commit_message': body})
    if result.get('merged') is not True:
        raise ValueError('GitHub did not merge the PR; check its current state')
    return result['sha']


def merge_request(api, event):
    request = command(api, event)
    number = event['issue']['number']
    draft_comment = preview(api, number, request)
    draft = load_draft(api, number, draft_comment)
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
