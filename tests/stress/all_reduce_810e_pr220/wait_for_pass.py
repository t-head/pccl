#!/usr/bin/env python3
"""Allow the follow-up only after the exact all-reduce run passed 100/100."""
import argparse
import json
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen

from stress import save_json, timestamp


def verify_summary(summary, pccl_revision, tests_revision):
    expected = {"requested_runs": 100, "attempted_runs": 100, "completed_runs": 100,
                "passed": 100, "failed": 0, "complete": True, "interrupted": False,
                "artifact_changed": False, "pccl_revision": pccl_revision,
                "pccl_tests_revision": tests_revision}
    for key, value in expected.items():
        if key not in summary or type(summary[key]) is not type(value) or summary[key] != value:
            raise ValueError(f"Predecessor summary does not satisfy {key}={value!r}")
    for key in ("failed_iterations", "target_hggc_failure_iterations",
                "hg_mem_map_invalid_argument_iterations", "timeout_iterations"):
        if summary.get(key) != []:
            raise ValueError(f"Predecessor has failures or missing evidence in {key}")
    runs = summary.get("runs", [])
    if len(runs) != 100 or [row.get("iteration") for row in runs] != list(range(1, 101)):
        raise ValueError("Predecessor does not contain exactly 100 ordered results")
    for row in runs:
        if (row.get("status") != "PASSED" or row.get("return_code") != 0
                or row.get("fatal_signatures") != [] or row.get("target_hggc_failure") is not False):
            raise ValueError("Predecessor contains a failed or invalid individual result")
    artifacts = summary.get("artifacts", {})
    for name in ("all_reduce_perf", "libpccl"):
        artifact = artifacts.get(name, {})
        before, after = artifact.get("sha256"), artifact.get("sha256_after")
        if not isinstance(before, str) or len(before) != 64 or before != after:
            raise ValueError(f"Predecessor artifact changed or is unverified: {name}")


def get_run(run_id):
    token = os.environ["GITHUB_TOKEN"]
    request = Request(f"https://api.github.com/repos/t-head/pccl/actions/runs/{run_id}", headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "pccl-conditional-sanity-stress",
    })
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def wait(run_id, expected_head, summary_path, pccl_revision, tests_revision,
         timeout, interval, output, get_state=get_run, sleep=time.sleep, clock=time.monotonic):
    deadline = clock() + timeout
    while clock() < deadline:
        try:
            state = get_state(run_id)
        except (OSError, ValueError) as error:
            print(f"Predecessor status temporarily unavailable: {type(error).__name__}", flush=True)
            sleep(min(interval, max(0, deadline - clock())))
            continue
        if state.get("head_sha") != expected_head or state.get("run_attempt") != 1:
            reason = "Predecessor source or attempt changed"
            break
        if state.get("status") != "completed":
            print(f"{timestamp()} predecessor {run_id}: {state.get('status')}; no PPU requested", flush=True)
            sleep(min(interval, max(0, deadline - clock())))
            continue
        if state.get("conclusion") != "success":
            reason = f"Predecessor ended with {state.get('conclusion')}; follow-up skipped"
            break
        try:
            summary = json.loads(Path(summary_path).read_text())
            verify_summary(summary, pccl_revision, tests_revision)
        except (OSError, ValueError, TypeError, KeyError) as error:
            reason = f"Predecessor pass evidence is invalid: {error}"
            break
        result = {"should_run": True, "reason": "Verified 100/100 all-reduce passes",
                  "predecessor_run_id": run_id, "predecessor_head": expected_head,
                  "pccl_revision": pccl_revision, "pccl_tests_revision": tests_revision,
                  "verified_at": timestamp()}
        save_json(output, result)
        return result
    else:
        reason = "Timed out waiting for predecessor; follow-up skipped"
    result = {"should_run": False, "reason": reason, "predecessor_run_id": run_id,
              "verified_at": timestamp()}
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pccl-revision", required=True)
    parser.add_argument("--tests-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=45000)
    parser.add_argument("--interval", type=float, default=60)
    args = parser.parse_args()
    if args.timeout <= 0 or args.interval <= 0:
        parser.error("timeout and interval must be positive")
    result = wait(args.run_id, args.expected_head, args.summary, args.pccl_revision,
                  args.tests_revision, args.timeout, args.interval, args.output)
    print(result["reason"], flush=True)
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write(f"should_run={str(result['should_run']).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
