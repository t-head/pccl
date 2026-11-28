"""Exercise the workflow's bounded APT fallback without network or root access."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
INSTALL = ROOT / ".github/workflows/scripts/pccl_m890p_coverage_apt.sh"
MIRRORS = ["default", "mirrors.aliyun.com", "mirrors.tuna.tsinghua.edu.cn", "mirrors.ustc.edu.cn"]
PACKAGES = ["gcc-13", "python3-venv"]

FAKE_APT = """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

args = sys.argv[1:]
options = {}
while args[0] == "-o":
    name, value = args[1].split("=", 1)
    options[name] = value
    del args[:2]
lists = Path(options["Dir::State::lists"])
source_list = options.get("Dir::Etc::sourcelist")
sources = Path(source_list).read_text().splitlines() if source_list else []
mirror = urlsplit(sources[0].split()[2]).hostname if sources else "default"
source_parts = options.get("Dir::Etc::sourceparts")
with open(os.environ["APT_TRACE"], "a") as log:
    print(json.dumps(dict(args=args, options=options, sources=sources, mirror=mirror,
                          timeout=os.environ["APT_TIMEOUT"], indexes=sorted(p.name for p in lists.iterdir()),
                          overrides_source_parts=bool(source_parts) and not any(Path(source_parts).iterdir()))),
          file=log)
if args[-1] == "update":
    if mirror != os.environ["APT_REFRESH_OK"]:
        (lists / "failed-index").touch()
        sys.exit("E: Hash Sum mismatch")
    (lists / "fresh-index").touch()
elif args[0] == "install":
    sys.exit(int(os.environ["APT_INSTALL_STATUS"]))
else:
    sys.exit(f"unexpected APT arguments: {args}")
"""


class CoverageAptTest(unittest.TestCase):
    def run_helper(self, refresh_ok="default", install_status=0):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            commands = {
                "apt-get": FAKE_APT,
                "timeout": '#!/bin/bash\nexport APT_TIMEOUT="$1 $2 $3"\nshift 3\nexec "$@"\n',
                "sleep": '#!/bin/bash\nprintf "%s\\n" "$@" >> "$APT_SLEEPS"\n',
            }
            for name, script in commands.items():
                executable = root / name
                executable.write_text(script)
                executable.chmod(0o755)
            trace, sleeps = root / "apt.jsonl", root / "sleeps"
            run = subprocess.run(["bash", INSTALL, *PACKAGES], capture_output=True, text=True, check=False,
                                 env={**os.environ, "PATH": f"{root}:{os.environ['PATH']}",
                                      "APT_TRACE": str(trace), "APT_SLEEPS": str(sleeps),
                                      "APT_REFRESH_OK": refresh_ok, "APT_INSTALL_STATUS": str(install_status)})
            calls = [json.loads(line) for line in trace.read_text().splitlines()]
            delays = sleeps.read_text().splitlines() if sleeps.exists() else []
        for call in calls:
            self.assertFalse(Path(call["options"]["Dir::State::lists"]).parent.exists())
        return run, calls, delays

    def assert_refreshes(self, calls, mirrors):
        self.assertEqual([call["mirror"] for call in calls], mirrors)
        self.assertEqual(len({call["options"]["Dir::State::lists"] for call in calls}), len(mirrors))
        for call in calls:
            self.assertEqual(call["args"], ["--error-on=any", "update"])
            self.assertEqual(call["timeout"], "-k 10s 120s")
            self.assertEqual(call["indexes"], [])
            self.assertEqual(call["options"]["Acquire::Retries"], "2")
            for protocol in ("http", "https"):
                self.assertEqual(call["options"][f"Acquire::{protocol}::Timeout"], "30")

    def assert_install_matches_refresh(self, refresh, install):
        self.assertEqual(install["args"], ["install", "-y", "--no-install-recommends", *PACKAGES])
        self.assertEqual(install["timeout"], "-k 10s 300s")
        self.assertEqual(install["options"], refresh["options"])
        self.assertEqual(install["sources"], refresh["sources"])
        self.assertEqual(install["indexes"], ["fresh-index"])

    def test_default_sources_succeed_without_fallback(self):
        run, calls, delays = self.run_helper()
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(delays, [])
        self.assert_refreshes(calls[:-1], MIRRORS[:1])
        self.assertNotIn("Dir::Etc::sourcelist", calls[0]["options"])
        self.assertNotIn("Dir::Etc::sourceparts", calls[0]["options"])
        self.assert_install_matches_refresh(*calls)

    def test_hash_mismatch_falls_back_to_signed_mirrors_in_order(self):
        run, calls, delays = self.run_helper(refresh_ok=MIRRORS[-1])
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stderr.count("Hash Sum mismatch"), 3)
        self.assertEqual(delays, ["15"] * 3)
        self.assert_refreshes(calls[:-1], MIRRORS)
        for call, mirror in zip(calls[1:-1], MIRRORS[1:]):
            self.assertTrue(call["overrides_source_parts"])
            self.assertEqual(call["sources"], [
                f"deb [signed-by=/usr/share/keyrings/ubuntu-archive-keyring.gpg] http://{mirror}/ubuntu/ "
                f"{suite} main restricted universe" for suite in ("noble", "noble-updates", "noble-security")
            ])
        self.assert_install_matches_refresh(*calls[-2:])

    def test_all_sources_fail_without_installing_from_stale_indexes(self):
        run, calls, delays = self.run_helper(refresh_ok="")
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("all fallback mirrors", run.stderr)
        self.assertEqual(delays, ["15"] * 3)
        self.assert_refreshes(calls, MIRRORS)

    def test_install_failure_does_not_restart_install_on_another_mirror(self):
        run, calls, delays = self.run_helper(install_status=100)
        self.assertEqual(run.returncode, 100, run.stderr)
        self.assertEqual(delays, [])
        self.assert_refreshes(calls[:-1], MIRRORS[:1])
        self.assert_install_matches_refresh(*calls)
