# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '.github/scripts'))
import prepare_squash as squash
from test_commit_messages import AUTHOR, MESSAGE

PROPOSAL = {'title': 'ci: simplify contribution checks',
            'body': '- allow follow-up commits during review\n- validate the final squash message before merging'}


class GitHubFixture:
    repository = 't-head/pccl'

    def __init__(self):
        self.pr = {'state': 'open', 'number': 98, 'title': 'Improve contribution checks', 'body': 'Refs #12',
                   'base': {'repo': {'full_name': self.repository}, 'ref': 'main', 'sha': 'b' * 40},
                   'head': {'sha': 'a' * 40}, 'commits': 2, 'changed_files': 1}
        self.files = [{'filename': 'example.py', 'status': 'modified', 'patch': '@@ -1 +1 @@\n-old\n+new',
                       'additions': 1, 'deletions': 1}]
        self.commits = [{'sha': str(i) * 40, 'commit': {'message': MESSAGE,
                         'author': {'name': AUTHOR[0], 'email': AUTHOR[1]}}} for i in (1, 2)]
        self.issue = {'number': 12}
        self.paths = []

    def get(self, path):
        self.paths.append(path)
        if '/issues/' in path:
            return copy.deepcopy(self.issue)
        return copy.deepcopy(self.pr)

    def pages(self, path):
        return copy.deepcopy(self.files if path.endswith('/files') else self.commits)


class SquashTests(unittest.TestCase):
    def setUp(self):
        self.api = GitHubFixture()

    def generate(self):
        return squash.generate(self.api, 98, lambda _: (copy.deepcopy(PROPOSAL), 'test-model'))

    def test_deduplicates_refs_and_signoffs(self):
        draft = self.generate()
        self.assertEqual(draft['message'].count('Refs #12'), 1)
        self.assertEqual(draft['message'].count('Signed-off-by:'), 1)
        squash.verify(self.api, draft)

    def test_multiple_authors_and_coauthors_are_preserved(self):
        second = self.api.commits[1]['commit']
        second['author'] = {'name': 'Second Author', 'email': 'second@example.org'}
        second['message'] = MESSAGE + 'Signed-off-by: Second Author <second@example.org>\nCo-authored-by: Collaborator <collaborator@example.org>\n'
        message = self.generate()['message']
        self.assertEqual(message.count('Signed-off-by:'), 2)
        self.assertIn('Co-authored-by: Collaborator <collaborator@example.org>', message)

    def test_preserves_closing_intent_and_issue_urls(self):
        self.api.commits[0]['commit']['message'] = MESSAGE.replace('Refs #12', 'Closes #12')
        self.api.commits[1]['commit']['message'] = MESSAGE.replace('Refs #12', 'Fixes https://github.com/t-head/other/issues/7')
        message = self.generate()['message']
        self.assertIn('Closes #12', message)
        self.assertIn('Fixes https://github.com/t-head/other/issues/7', message)
        self.assertIn('/repos/t-head/other/issues/7', self.api.paths)

    def test_rejects_missing_metadata_in_any_original_commit(self):
        self.api.commits[1]['commit']['message'] = MESSAGE.replace('Refs #12', '')
        with self.assertRaises(ValueError):
            self.generate()

    def test_rejects_pr_used_as_issue(self):
        self.api.issue['pull_request'] = {}
        with self.assertRaises(ValueError):
            self.generate()

    def test_rejects_incomplete_lists(self):
        for field in ('changed_files', 'commits'):
            with self.subTest(field=field):
                api = GitHubFixture()
                api.pr[field] += 1
                with self.assertRaises(ValueError):
                    squash.snapshot(api, 98)

    def test_rejects_binary_and_truncated_diffs(self):
        for file in [dict(self.api.files[0], patch=''), dict(self.api.files[0], additions=2)]:
            with self.subTest(file=file):
                self.api.files = [file]
                with self.assertRaises(ValueError):
                    squash.snapshot(self.api, 98)

    def test_rejects_closed_pr(self):
        self.api.pr['state'] = 'closed'
        with self.assertRaises(ValueError):
            self.generate()

    def test_rejects_changed_head_during_generation(self):
        def ask(_):
            self.api.pr['head']['sha'] = 'c' * 40
            return PROPOSAL, 'test-model'
        with self.assertRaisesRegex(ValueError, 'changed during generation'):
            squash.generate(self.api, 98, ask)

    def test_rejects_stale_head_base_title_body_or_diff(self):
        for field in ('head', 'base', 'title', 'body', 'diff'):
            with self.subTest(field=field):
                self.api = GitHubFixture()
                draft = self.generate()
                if field in ('head', 'base'):
                    self.api.pr[field]['sha'] = 'c' * 40
                elif field == 'diff':
                    self.api.files[0]['patch'] = '@@ -1 +1 @@\n-old\n+changed'
                else:
                    self.api.pr[field] = 'Changed'
                with self.assertRaisesRegex(ValueError, 'changed'):
                    squash.verify(self.api, draft)

    def test_rejects_tampered_final_message(self):
        draft = self.generate()
        draft['message'] = draft['message'].replace('Refs #12', 'Closes #12')
        with self.assertRaises(ValueError):
            squash.verify(self.api, draft)

    def test_rejects_model_metadata_and_control_characters(self):
        data = squash.snapshot(self.api, 98)
        for text in ['Signed-off-by: Fake <fake@example.org>', 'By Signed-off-by: Fake <fake@example.org>', 'Refs #99',
                     'https://github.com/t-head/pccl/issues/99', 'Fixes: #99', '\x1b[31mred', '\rhidden']:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    squash.final_message(dict(PROPOSAL, body=text), data)

    def test_rejects_invalid_model_schema(self):
        data = squash.snapshot(self.api, 98)
        for proposal in [None, {'title': 'Title'}, dict(PROPOSAL, extra=True),
                         dict(PROPOSAL, title='a\nb'), dict(PROPOSAL, body=[]), dict(PROPOSAL, body='')]:
            with self.subTest(proposal=proposal):
                with self.assertRaises(ValueError):
                    squash.final_message(proposal, data)

    def test_oversized_input_is_not_silently_truncated(self):
        data = squash.snapshot(self.api, 98)
        data['body'] = 'a' * 120_000
        with self.assertRaises(ValueError):
            squash.prompt_for(data)

    def test_summary_escapes_message_and_tests_do_not_write_job_summary(self):
        draft = self.generate()
        draft['message'] += '<script>alert(1)</script>'
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            squash.write_output(draft, Path(directory))
            summary = (Path(directory) / 'summary.md').read_text()
            self.assertNotIn('<script>', summary)
            self.assertIn('&lt;script&gt;', summary)
            self.assertEqual(json.loads((Path(directory) / 'proposal.json').read_text())['head'], draft['head'])

    def test_model_cannot_execute_tools(self):
        event = json.dumps({'type': 'tool.execution_start'})
        with patch.dict(os.environ, {'COPILOT_CLI': '/example/copilot', 'GITHUB_TOKEN': 'test'}), \
                patch.object(squash.subprocess, 'run') as run:
            run.return_value.returncode = 0
            run.return_value.stdout = event
            with self.assertRaisesRegex(ValueError, 'must not execute tools'):
                squash.ask_copilot('Example')


if __name__ == '__main__':
    unittest.main()
