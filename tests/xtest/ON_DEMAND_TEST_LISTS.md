# On-demand public test lists

This path generates Case inputs in the public `pccl` repository at
workflow prepare time. It uses only the Python standard library and the existing
public CSV consumer; xtest and mmengine are neither installed nor imported.

The fourteen Runner-backed daily definitions are published by the full source
JSON exporter at source revision `80738685d8b2e2066223119594805e5255f4a8c8`.
`test_lists/source-export.json` records their exact 8,899-Case, 34-Suite
real-Runner comparison and SHA-256 digests. CSV is not an input or output of
that publication step.

## Maintained inputs

- `test_lists/op_perf_shm.json`: nine operations and rank layouts for 1/2/4/8
  nodes, producing 54 Cases.
- `test_lists/postsil_daily_shm.json`: operation, algorithm, protocol, dtype,
  size, iteration and layout axes, producing 3,938 combinations plus two
  explicit special Cases.
- `test_lists/pccl_fun_rdma_shm.json`: 4,090 multi-node combinations, including
  LL128, 4nics, host, socket and ibsplit variants.
- `test_lists/register_test_shm.json` and
  `test_lists/sanity_m890p_register.json`: register layouts and special groups.
- `test_lists/sym_test_shm.json`, `test_lists/f8_test_shm.json`,
  `test_lists/device_api_shm.json`, `test_lists/local_register_perf_shm.json`,
  `test_lists/symetric_perf_shm.json`, `test_lists/copyengine_perf_shm.json`,
  `test_lists/op_perf_shm_2_rack.json`, `test_lists/op_perf_rdma_shm.json`, and
  `test_lists/commabort_test_shm.json`: the remaining Runner families.
- `test_lists/suites.json`: Plan/Suite identity, nodes, cards, timeout and shared
  environment references.
- `test_lists/envs/`: shared public runtime environments.
- `test_lists/source-export.json`: full source selection, per-family row
  digests, definition hashes and the combined Case/Suite equivalence manifest.

Generated CSV is a temporary execution artifact, not a maintained input. All
daily datasets use this generation path; the source-sanity CSV is independent
and remains unchanged.

## Generate locally

```bash
python3 -S tests/xtest/generate_cases.py \
  --definition tests/xtest/test_lists/op_perf_shm.json \
  --definition tests/xtest/test_lists/postsil_daily_shm.json \
  --suites tests/xtest/test_lists/suites.json \
  --env-root tests/xtest/test_lists/envs \
  --profile full \
  --source-revision 1111111111111111111111111111111111111111 \
  --runner-revision 2222222222222222222222222222222222222222 \
  --output /tmp/pccl-generated-cases
```

Use `--profile smoke` to generate two Cases, one from each family. Family
definitions also support their own more specific profiles and `--nodes` filter.
The output must not already exist.

The two-family example is a subset. The workflow's `full` selection
passes all fourteen definitions to the same command.

## Workflow datasets

The workflow supports two `workflow_dispatch` choices:

- `smoke`: two Cases in one execution group.
- `full`: all 8,899 daily Cases in 34 Suites. The current
  1/2/4/8-node resource policy selects 8,890 Cases in 25 groups and records
  only the 9 sixteen-node Cases as excluded. Two-rack Suites are isolated from
  ordinary same-resource Suites and use `rack_policy=spread` with `rack_count=2`.
  Ordinary multi-node groups use `rack_policy=same`, keeping every worker in one
  automatically selected rack without binding a physical rack name.

The 22:00 Asia/Shanghai schedule selects `full`, which uses the same JSON
generation path as manual dispatch. Monday-Thursday allows 1/2/4/8 nodes and
selects 8,890 Cases in 25 groups with 9 exclusions. Friday allows 1/2/4/8/16
nodes and selects all 8,899 Cases in 26 groups. Manual dispatch keeps the
weekday default unless `allowed_nodes` is explicitly overridden. No legacy
daily CSV dataset remains.

The per-group `cluster-run` queue-and-execution timeout is 720 minutes for the
Monday-Thursday schedule and 2,160 minutes (36 hours) for the Friday schedule.
Manual dispatch uses `execution_timeout_minutes`, which defaults to 720 and can
be set explicitly. The execute Job allows 30 additional minutes for setup and
cleanup. Suite and Case timeouts are unchanged.

Each generated bundle contains `generation.json` with definition and Suite
hashes, the selected profile, and the exact Case-key hash. The workflow copies
it into the Case-groups artifact. `export.json` retains the existing consumer
schema and records `xtest_version=not-used` explicitly. Source provenance is
retained separately from the public `pccl` revision.

## Performance Excel report

Prepare also writes `performance-cases.json` before resource-policy grouping.
The full selection contains 168 performance-related Cases, including ordinary
`op_perf` Cases whose `check` remains `exit_code` and Cases later excluded by the
node policy. This preserves complete report coverage rather than only the
successfully executed subset.

After group reconciliation, the summary job uses `run_cases.parse_performance`
to read the retained `raw.log` files and generate
`PCCL_M890P_daily_perf_<run-id>.xlsx`. Ordinary `exit_code` Cases contribute
performance data without gaining a performance gate. Execution failures,
missing results, unavailable tables, partial curves and missing log rows remain
visible as separate execution and data statuses.

Only `PASSED` Cases contribute valid performance points and detail rows.
Failed Cases remain visible in coverage, but their partial output is not counted
as valid performance evidence. Duplicate sizes, size zero and `N/A` validation
markers are preserved. Curve completeness follows the expected measurement
count per layout rather than a Case-plus-size uniqueness assumption.

The workbook contains an overview, per-topology tables, raw detail, Case
coverage and run metadata. It is uploaded as the independent
`m890p-daily-performance-report` artifact with 90-day retention. Raw logs
remain in the run-owned NAS `results/` tree; the Excel report does not replace
that diagnostic evidence or the existing summary artifact.

XlsxWriter 3.2.5 is pinned with a SHA-256 digest in the existing dependency
artifact and imported directly from its wheel by the summary job; pip is not
required. Report generation is part of the final failure summary, so a missing
required report cannot pass silently.

## Validation boundary

Tests compare every generated row with the source-export digests: all 8,899
Cases and all 34 Suite definitions match,
including commands, checks, Suite resources, shared environments and explicit
special Cases. Two legacy `device_api` Plan/Suite identities are explicitly
normalized to the current source names; their 52 commands and resources remain
unchanged. The existing grouping and public input consumer accept the generated
bundle and reproduce the same 8,890 selected Cases, 25 groups and 9 resource
exclusions.

All current daily full Suites are covered, and no generated Python Case files
are produced. Hardware validation is limited to explicitly filtered datasets;
no unfiltered full hardware run has been started.

Workflow run `43585016370` consumed source-exported families during prepare,
passed both single-node hardware Cases, retained source provenance in the
Case-groups artifact, and completed exact reconciliation and NAS cleanup.
