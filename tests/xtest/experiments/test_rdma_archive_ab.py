"""Local contracts for the diagnostic experiment, without hardware or NAS."""
import importlib.util
import os
import subprocess
import tarfile
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).with_name('rdma_archive_ab.py')
spec = importlib.util.spec_from_file_location('rdma_archive_ab', MODULE_PATH)
ab = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ab)


class ArchiveExtractionTests(unittest.TestCase):
    def test_extract_and_verify_both_sources_with_scoped_trust(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / 'shared' / 'good-first'
            archive = Path(temporary) / 'output_standard.tar.gz'
            revision = ab.ARCHIVES['good'][0]
            runner_revision = ab.artifact_cache.PCCL_TESTS_REVISION
            with patch.object(ab.subprocess, 'run') as extract, patch.object(
                    ab.subprocess, 'check_output', side_effect=[revision+'\n', runner_revision+'\n']) as git:
                build = ab.extract_verified_build(archive, directory, revision)
            self.assertEqual(build, directory / 'output_standard')
            extract.assert_called_once_with(
                ['tar', '--no-same-owner', '--no-same-permissions', '-xzf', str(archive),
                 '-C', str(directory)], check=True)
            self.assertEqual(git.call_count, 2)
            for call, repository in zip(git.call_args_list, ('pccl', 'pccl-tests')):
                path = build / repository
                self.assertEqual(call.args[0], ['git', '-c', f'safe.directory={path}',
                                               '-C', str(path), 'rev-parse', 'HEAD'])
                self.assertNotIn('*', call.args[0])

    def test_real_archive_is_readable_and_keeps_executables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source' / 'output_standard'
            revisions = []
            for name in ('pccl', 'pccl-tests'):
                repository = source / name
                repository.mkdir(parents=True)
                subprocess.run(['git', 'init', '-q', str(repository)], check=True)
                subprocess.run(['git', '-C', str(repository), '-c', 'user.name=Test',
                                '-c', 'user.email=test@example.invalid', '-c', 'core.hooksPath=/dev/null',
                                'commit', '-q', '--allow-empty', '-m', 'Fixture'], check=True)
                revisions.append(subprocess.check_output(
                    ['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip())
            binary = source / 'pccl-tests' / 'build' / 'probe'
            binary.parent.mkdir()
            binary.write_text('#!/bin/sh\nexit 0\n')
            binary.chmod(0o755)
            archive = root / 'archive.tar.gz'
            def archived_owner(info):
                info.uid = info.gid = 12345
                info.uname = info.gname = 'historical-builder'
                return info
            with tarfile.open(archive, 'w:gz') as tar:
                tar.add(source, arcname='output_standard', filter=archived_owner)
            previous_umask = os.umask(0o022)
            try:
                with patch.object(ab.artifact_cache, 'PCCL_TESTS_REVISION', revisions[1]):
                    build = ab.extract_verified_build(archive, root / 'shared', revisions[0])
            finally:
                os.umask(previous_umask)
            extracted = build / 'pccl-tests' / 'build' / 'probe'
            self.assertEqual(extracted.stat().st_uid, os.getuid())
            self.assertEqual(extracted.stat().st_mode & 0o777, 0o755)
            self.assertEqual((root / 'shared').stat().st_mode & 0o777, 0o755)

    def test_rejects_wrong_pccl_revision(self):
        self.assert_revision_rejected(['0'*40])

    def test_rejects_wrong_runner_revision(self):
        self.assert_revision_rejected([ab.ARCHIVES['good'][0], '0'*40])

    def assert_revision_rejected(self, revisions):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(ab.subprocess, 'run'), patch.object(
                    ab.subprocess, 'check_output', side_effect=revisions):
                with self.assertRaisesRegex(ValueError, 'Extracted source mismatch'):
                    ab.extract_verified_build(Path(temporary)/'archive',
                                              Path(temporary)/'shared', ab.ARCHIVES['good'][0])

    def test_all_three_rounds_use_the_validated_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = type('Args', (), {'root': root, 'cache': root/'cache',
                                    'hostfile': root/'hosts', 'interface': 'eth0'})()
            selected = {'cases': []}
            results = [{'name': name, 'status': 'PASSED'} for name, _ in ab.cases()]
            with patch.object(ab.artifact_cache, 'validate', return_value=root/'archive'), \
                    patch.object(ab.artifact_cache, 'read_record', return_value={}), \
                    patch.object(ab, 'extract_verified_build', side_effect=lambda a,p,r:p/'output_standard') as extract, \
                    patch.object(ab.run_cases, 'load_inputs', return_value=selected), \
                    patch.object(ab.run_cases, 'run_all', return_value=results):
                self.assertEqual(ab.run(args), 0)
            self.assertEqual(extract.call_count, 3)
            self.assertEqual([c.args[2] for c in extract.call_args_list],
                             [ab.ARCHIVES['good'][0], ab.ARCHIVES['failed'][0], ab.ARCHIVES['good'][0]])


if __name__ == '__main__':
    unittest.main()
