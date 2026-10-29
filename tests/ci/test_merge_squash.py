# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.

import copy
import json
import subprocess
import unittest
from unittest.mock import patch

from test_prepare_squash import GitHubFixture, PROPOSAL
import merge_squash as merge
import prepare_squash as squash


class MergeFixture(GitHubFixture):
    token = 'test-token'

    def __init__(self):
        super().__init__()
        self.permission = 'write'
        self.command = {'id': 456, 'body': '/merge',
                        'user': {'id': 1, 'login': 'contributor', 'type': 'User'},
                        'created_at': '2026-10-29T01:01:00Z', 'updated_at': '2026-10-29T01:01:00Z',
                        'issue_url': 'https://api.github.com/repos/t-head/pccl/issues/98'}
        self.event = {'action': 'created', 'repository': {'full_name': self.repository},
                      'issue': {'number': 98, 'pull_request': {'url': 'pulls/98'}},
                      'comment': copy.deepcopy(self.command)}
        self.draft = squash.generate(self, 98, lambda _: PROPOSAL)
        self.comments = [{'id': 123, 'body': squash.comment_body(self.draft, 99),
                          'updated_at': '2026-10-29T01:00:00Z',
                          'user': {'type': 'Bot', 'login': 'github-actions[bot]'}}]
        self.run = {'path': '.github/workflows/prepare-squash.yml', 'status': 'completed',
                    'conclusion': 'success', 'event': 'pull_request', 'head_branch': 'main',
                    'repository': {'full_name': self.repository},
                    'head_repository': {'full_name': self.repository}}
        self.ready = {'state': 'OPEN', 'isDraft': False, 'baseRefName': 'main',
                      'headRefOid': self.draft['head'], 'baseRefOid': self.draft['base'],
                      'reviewDecision': 'APPROVED', 'mergeStateStatus': 'CLEAN'}

    def get(self, path):
        if path.endswith('/permission'):
            return {'permission': self.permission}
        if path.endswith('/issues/comments/456'):
            return copy.deepcopy(self.command)
        if path.endswith('/actions/runs/99'):
            return copy.deepcopy(self.run)
        return super().get(path)

    def cli(self, *args):
        assert args[0] == 'pr', 'Merging must not download artifacts'
        return json.dumps(self.ready).encode()


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.api = MergeFixture()

    def attempt(self):
        return merge.merge_request(self.api, self.api.event)

    def assert_rejected(self):
        with patch.object(merge, 'merge_message') as write:
            with self.assertRaises(ValueError):
                self.attempt()
            write.assert_not_called()

    def test_merges_the_exact_verified_draft(self):
        with patch.object(merge, 'merge_message', return_value='c' * 40) as write:
            self.assertEqual(self.attempt(), 'c' * 40)
        write.assert_called_once_with(self.api, 98, self.api.draft)

    def test_requires_repository_write_access(self):
        for permission in ('read', 'triage', 'none'):
            with self.subTest(permission=permission):
                self.api.permission = permission
                self.assert_rejected()
        self.api.permission = 'admin'
        with patch.object(merge, 'merge_message', return_value='c' * 40):
            self.attempt()

    def test_rejects_non_commands_bots_and_edited_commands(self):
        original = copy.deepcopy(self.api.command)
        for change in ({'body': 'please /merge'}, {'updated_at': '2026-10-29T01:02:00Z'},
                       {'user': {'id': 1, 'login': 'some-bot', 'type': 'Bot'}}):
            with self.subTest(change=change):
                self.api.command = {**original, **change}
                self.assert_rejected()

    def test_refuses_retargeted_or_non_pr_events(self):
        for change in ({'repository': {'full_name': 'other/repo'}},
                       {'action': 'edited'}, {'issue': {'number': 98}}):
            with self.subTest(change=change):
                self.api = MergeFixture()
                self.api.event.update(change)
                self.assert_rejected()

    def test_requires_one_bot_preview_displayed_before_the_command(self):
        original = copy.deepcopy(self.api.comments)
        for comments in ([], original * 2,
                         [{**original[0], 'updated_at': self.api.command['created_at']}],
                         [{**original[0], 'user': {'login': 'contributor', 'type': 'User'}}]):
            with self.subTest(comments=comments):
                self.api.comments = comments
                self.assert_rejected()

    def test_refuses_untrusted_or_unsuccessful_generation_runs(self):
        original = copy.deepcopy(self.api.run)
        for change in ({'path': '.github/workflows/other.yml'}, {'conclusion': 'failure'},
                       {'event': 'push'}, {'status': 'in_progress'},
                       {'head_repository': {'full_name': 'contributor/fork'}},
                       {'event': 'workflow_dispatch', 'head_branch': 'feature'}):
            with self.subTest(change=change):
                self.api.run = {**original, **change}
                self.assert_rejected()

    def test_refuses_incomplete_records_wrong_prs_and_changed_messages(self):
        original = self.api.comments[0]['body']
        for body in (original + 'An edited message.',
                     original.replace('<!-- draft:', '<!-- removed:'),
                     original.replace('"number":98', '"number":19'),
                     original.replace('Refs #12', 'Closes #12')):
            with self.subTest(body=body):
                self.api.comments[0]['body'] = body
                self.assert_rejected()

    def test_comment_round_trip_preserves_text_and_hides_verification_data(self):
        proposal = {**PROPOSAL, 'body': '- Handle -->, <tags>, & and Unicode λ as plain text.'}
        self.api.draft['proposal'] = proposal
        self.api.draft['message'] = squash.final_message(proposal, squash.snapshot(self.api, 98))
        body = squash.comment_body(self.api.draft, 99)
        self.api.comments[0]['body'] = body
        context = body.split('<!-- draft: ', 1)[1].split(' -->', 1)[0]
        self.assertNotIn('<', context)
        self.assertNotIn('>', context)
        self.assertEqual(merge.load_draft(self.api, 98, self.api.comments[0]), self.api.draft)
        with patch.object(merge, 'merge_message', return_value='c' * 40):
            self.attempt()

    def test_refuses_stale_pr_inputs_and_unmet_merge_requirements(self):
        original = copy.deepcopy(self.api.ready)
        for change in ({'headRefOid': 'c' * 40}, {'baseRefOid': 'c' * 40},
                       {'baseRefName': 'release'}, {'isDraft': True}, {'state': 'CLOSED'},
                       {'reviewDecision': 'REVIEW_REQUIRED'}, {'mergeStateStatus': 'BLOCKED'},
                       {'mergeStateStatus': 'UNKNOWN'}):
            with self.subTest(change=change):
                self.api.ready = {**original, **change}
                self.assert_rejected()
        self.api.ready = original
        self.api.pr['body'] += '\nNew information'
        self.assert_rejected()

    def test_rechecks_reviews_before_the_merge_write(self):
        original = squash.verify
        def verify(*args):
            result = original(*args)
            self.api.ready['reviewDecision'] = 'REVIEW_REQUIRED'
            return result
        with patch.object(squash, 'verify', side_effect=verify):
            self.assert_rejected()

    def test_rest_merge_binds_head_and_preserves_refs_and_dco_once(self):
        api = squash.GitHub(self.api.repository, self.api.token)
        with patch.object(squash.subprocess, 'run') as send:
            send.return_value.stdout = json.dumps({'merged': True, 'sha': 'c' * 40})
            self.assertEqual(merge.merge_message(api, 98, self.api.draft), 'c' * 40)
        self.assertEqual(send.call_args.args[0], ['gh', 'api', '/repos/t-head/pccl/pulls/98/merge',
                                                '--method', 'PUT', '--input', '-'])
        payload = json.loads(send.call_args.kwargs['input'])
        self.assertEqual(payload['sha'], self.api.draft['head'])
        self.assertEqual(payload['merge_method'], 'squash')
        message = payload['commit_title'] + '\n\n' + payload['commit_message']
        self.assertEqual(message, self.api.draft['message'].rstrip('\n'))
        self.assertEqual(message.count('Refs #12'), 1)
        self.assertEqual(message.count('Signed-off-by:'), 1)

    def test_refuses_github_merge_denials(self):
        api = squash.GitHub(self.api.repository, self.api.token)
        for code in (403, 405, 409):
            with self.subTest(code=code):
                error = subprocess.CalledProcessError(1, ['gh'], stderr=f'HTTP {code}')
                with patch.object(squash.subprocess, 'run', side_effect=error), self.assertRaises(ValueError):
                    merge.merge_message(api, 98, self.api.draft)
        with patch.object(squash.subprocess, 'run') as send:
            send.return_value.stdout = '{"merged": false}'
            with self.assertRaises(ValueError):
                merge.merge_message(api, 98, self.api.draft)


if __name__ == '__main__':
    unittest.main()
