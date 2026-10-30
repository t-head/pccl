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
    comment = api.request(f"/repos/{api.repository}/issues/comments/{expected['id']}")
    if (comment['user']['type'] != 'User' or comment['user']['id'] != expected['user']['id'] or
            comment['body'] != '/merge' or comment['body'] != expected['body'] or
            comment['created_at'] != comment['updated_at'] or
            comment['created_at'] != expected['created_at'] or
            comment['issue_url'] != f'https://api.github.com/repos/{api.repository}/issues/{number}'):
        raise ValueError('The command changed or is not a new human merge request')
    login = comment['user']['login']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9-]{0,38}', login):
        raise ValueError('Invalid command author')
    permission = api.request(f'/repos/{api.repository}/collaborators/{login}/permission')
    if permission['permission'] not in {'write', 'admin'}:
        raise ValueError('The command author needs repository Write, Maintain or Admin access')


def preview(api, number, request):
    comments = squash.preview_comments(api, number)
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
    run = api.request(f'/repos/{api.repository}/actions/runs/{run_id}')
    if (run['path'] != '.github/workflows/prepare-squash.yml' or
            run['repository']['full_name'] != api.repository or
            run['status'] != 'completed' or run['conclusion'] != 'success' or
            run['event'] not in {'pull_request', 'workflow_dispatch'} or
            (run['event'] == 'pull_request' and
             (run['head_repository']['full_name'] != api.repository or run['head_sha'] != draft['head'])) or
            (run['event'] == 'workflow_dispatch' and run['head_branch'] != 'main')):
        raise ValueError('The preview must reference a successful message preparation workflow')
    return draft


def merge_message(api, number, draft):
    title, body = draft['message'].rstrip('\n').split('\n\n', 1)
    result = api.request(f'/repos/{api.repository}/pulls/{number}/merge', 'PUT',
                         {'sha': draft['head'], 'merge_method': 'squash',
                          'commit_title': title, 'commit_message': body})
    if result.get('merged') is not True:
        raise ValueError('GitHub did not merge the PR; check its current state')
    return result['sha']


def merge_request(api, event):
    number = event['issue']['number']
    draft_comment = preview(api, number, event['comment'])
    draft = load_draft(api, number, draft_comment)
    if draft['target'] != 'main':
        raise ValueError('Only PRs targeting main can be merged')
    # Authorize the unchanged command near the write. Publishing shares this
    # job's concurrency group, so re-read the selected preview by ID.
    command(api, event)
    current_comment = api.request(f"/repos/{api.repository}/issues/comments/{draft_comment['id']}")
    if any(current_comment[key] != draft_comment[key] for key in ('id', 'body', 'updated_at')):
        raise ValueError('The displayed draft changed; review it and send /merge again')
    squash.verify(api, draft)
    return merge_message(api, number, draft)


def error_detail(api, error):
    """Keep diagnostics useful without exposing the workflow token."""
    detail = ' '.join(str(error).replace(api.token, '[redacted]').split())[:500]
    print(f'{type(error).__name__}: {detail}', file=sys.stderr)
    return detail


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
        detail = error_detail(api, error)
        result = 'Could not merge: ' + (detail if isinstance(error, ValueError)
                                      else 'Verification failed; see the merge workflow logs')
        status = 1
    else:
        # GITHUB_TOKEN merges do not trigger push workflows. Explicit dispatch
        # is supported, and starts post-merge CI without changing PR requirements.
        try:
            api.request(f'/repos/{api.repository}/actions/workflows/pr-test-ppu.yml/dispatches',
                        'POST', {'ref': 'main'})
            result += ' Main CI has been requested.'
        except Exception as error:
            error_detail(api, error)
            result += ' Main CI could not be started; run pr-test-ppu.yml on main manually.'
            status = 1
    print(result)
    try:
        api.request(f"/repos/{api.repository}/issues/{event['issue']['number']}/comments",
                    'POST', {'body': result})
    except Exception as error:
        error_detail(api, error)
        print('Could not publish the result comment; check the PR and workflow logs.', file=sys.stderr)
    return status


if __name__ == '__main__':
    sys.exit(main())
