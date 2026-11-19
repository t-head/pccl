# SPDX-FileCopyrightText: Copyright (c) 2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / '.github/workflows/scripts/install_hpn_sdk.sh'
URL = ('https://zone-ci.oss-cn-hangzhou.aliyuncs.com/hpn-sdk/2.1.3/'
       'hpnsdk2.1.3-ubuntu2404-202611121836-x86_64.bin')


class HpnSdkInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.calls = self.root / 'calls'
        self.wget = self.root / 'wget-log'
        self.install = self.root / 'install-log'
        (self.bin / 'wget').write_text(
            '#!/bin/bash\n'
            'printf "%s\\n" "$*" > "$WGET_LOG"\n'
            'while (( $# )); do if [[ $1 == -O ]]; then output=$2; break; fi; shift; done\n'
            'printf "installer\\n" > "$output"\n'
            'exit "${WGET_RC:-0}"\n')
        (self.bin / 'bash').write_text(
            '#!/bin/bash\n'
            'printf "%s\\n" "$*" > "$INSTALL_LOG"\n'
            'exit "${INSTALL_RC:-0}"\n')
        for path in self.bin.iterdir():
            path.chmod(0o755)

    def tearDown(self):
        self.temporary.cleanup()

    def run_install(self, url, extra_environment=None):
        environment = dict(os.environ, PATH=str(self.bin) + ':/usr/bin:/bin',
                           HPN_SDK_URL=url, WGET_LOG=str(self.wget), INSTALL_LOG=str(self.install))
        environment.update(extra_environment or {})
        return subprocess.run(['/bin/bash', str(SCRIPT)], env=environment,
                              capture_output=True, text=True)

    def test_empty_url_preserves_image_builtin_sdk(self):
        result = self.run_install('')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('image-builtin', result.stdout)
        self.assertFalse(self.wget.exists())
        self.assertFalse(self.install.exists())

    def test_explicit_url_downloads_and_installs_once(self):
        result = self.run_install(URL)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(URL, result.stdout)
        self.assertIn(URL, self.wget.read_text())
        self.assertIn('--tries=3', self.wget.read_text())
        self.assertIn('install --force', self.install.read_text())
        self.assertEqual(self.wget.read_text().count(URL), 1)
        self.assertEqual(self.install.read_text().count('install --force'), 1)

    def test_untrusted_or_non_https_urls_fail_before_download(self):
        for url in ('http://zone-ci.oss-cn-hangzhou.aliyuncs.com/hpn-sdk/sdk.bin',
                    'https://example.invalid/hpn-sdk/sdk.bin',
                    'https://zone-ci.oss-cn-hangzhou.aliyuncs.com/hpn-sdk/sdk.txt'):
            with self.subTest(url=url):
                result = self.run_install(url)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('HPN_SDK_URL', result.stderr)
        self.assertFalse(self.wget.exists())
        self.assertFalse(self.install.exists())

    def test_install_failure_does_not_fallback_to_image(self):
        result = self.run_install(URL, {'INSTALL_RC': '9'})
        self.assertEqual(result.returncode, 9)
        self.assertNotIn('image-builtin', result.stdout)

    def test_download_failure_does_not_run_installer_or_fallback(self):
        result = self.run_install(URL, {'WGET_RC': '7'})
        self.assertEqual(result.returncode, 7)
        self.assertNotIn('image-builtin', result.stdout)
        self.assertFalse(self.install.exists())


if __name__ == '__main__':
    unittest.main()
