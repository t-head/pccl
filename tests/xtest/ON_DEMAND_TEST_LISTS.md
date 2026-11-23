# On-demand public test lists

This path generates Case inputs in the public `pccl` repository at
workflow prepare time. It uses only the Python standard library and the existing
public CSV consumer; xtest and mmengine are neither installed nor imported.

The original fourteen Runner-backed daily definitions were published by the
full source JSON exporter at source revision
`80738685d8b2e2066223119594805e5255f4a8c8`. The native unit family was captured
at `cefc8624ec1a2b58bc54e76eedeefae90046d84f` and is identified separately in
`source-export.json.additional_source_revisions`.
`test_lists/source-export.json` records their exact 10,193-Case, 35-Suite
real-Runner comparison and SHA-256 digests. CSV is not an input or output of
that publication step.

## Maintained inputs

- `test_lists/op_perf_shm.json`: nine operations and rank layouts for 1/2/4/8
  nodes, producing 54 Cases.
- `test_lists/postsil_daily_shm.json`: operation, algorithm, protocol, dtype,
  size, iteration and layout axes, producing 3,938 combinations plus two
  explicit special Cases.
- `test_lists/postsil_unit_test_shm.json`: 1,252 direct `collPcclUnitTest`
  combinations and 42 `hggcGraphUnitTest` configurations from the native unit
  Suite, plus two smoke Cases. The graph names are explicitly normalized from
  the source `cudaGraphUnitTest`/`cuda_graph_test_configs` spelling to the
  public binary and directory names.
  Source `--devices` lists are preserved through `HGGC_VISIBLE_DEVICES`;
  `--auto-detect` is removed because public native binaries do not accept it.
  Three `use_ext_kernel` configurations are explicitly excluded as unsupported.
- `test_lists/postsil_unit_test_810e.json`: the dedicated 810E native unit
  export with 1,252 `collPcclUnitTest` and 45 supported `hggcGraphUnitTest`
  Cases. `test_lists/all_gather_unit_test_810e.json` adds three all_gather
  configurations. `test_lists/postsil_unit_test_810e_smoke.json` selects one
  collective and one graph Case for two-card PR validation.
- `test_lists/source-export-810e.json`: dedicated 810E source revision,
  per-family and combined row digests, and 346 explicit exclusions: 308
  `p2pBandwidthLatencyTest`, 35 `alltoallvTest`, and three `use_ext_kernel`
  configurations whose public binaries or options are unavailable.
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
passes all fifteen definitions to the same command.

## Workflow datasets

The workflow supports two `workflow_dispatch` choices:

- `smoke`: four Cases in one execution group, including one collective and one
  graph native unit Case.
- `full`: all 10,193 daily Cases in 35 Suites. The current
  1/2/4/8-node resource policy selects 10,184 Cases in 28 groups and records
  only the 9 sixteen-node Cases as excluded. Two-rack Suites are isolated from
  ordinary same-resource Suites and use `rack_policy: spread` with
  `rack_count: 2`. Ordinary M890P multi-node groups use `rack_policy: same`
  for automatic same-rack placement. OAM-810E has no rack constraint.

These counts describe the preserved M890P baseline. Daily preparation also
applies `platforms/oam810e_eic.json` to the supported shared families and adds
7,666 full-selection Cases after excluding all 36 FP8 Cases and 24 GIN Cases
inherited from the shared device_api family. The dedicated 18-Case 810E GIN3
selection is deferred until GIN communicator support is validated on the target
environment; M890P GIN coverage is unchanged. Combined dual-platform generation
produces 17,859 Cases, selects 17,850 in 53 groups, and retains each group's
platform and `node_selector`. Dual-platform smoke has nine Cases in three groups, including
one 810E collective and one graph native unit Case.

The 810E allocation is derived from actual ranks per node rather than always
requesting 16 cards. MPI rank counts are divided by the number of nodes;
CommAbort multiplies its MPI processes by the thread and GPU counts; native
`num_gpus` and graph `num_threads * num_gpus` values determine their allocation.
Explicit device selections also reserve the span through the highest
local index, so graph `devices 0,3` requests four cards and
retains its topology instead of being compressed to two cards.
Every 810E allocation is then rounded up to the next power of two, so only 1,
2, 4, 8 or 16 cards are requested per node.
Grouping separates different rank counts before applying the 500-Case shard
limit. M890P keeps its existing 28 groups and allocation behavior.
At execution on both M890P and 810E, Case-local device indices are mapped to the
physical PPU list in
`NVIDIA_VISIBLE_DEVICES`, including explicit `HGGC_VISIBLE_DEVICES` and
`--devices` selections. Each worker validates its own allocation as a finite,
unique list of numeric indices before mapping; `all`, malformed entries and
duplicates (including equivalent values such as `9,09`) fail before launch.
The effective list is forwarded through MPI so parallel
Pods cannot silently fall back to the same host device 0/1. Out-of-range
selections fail before the benchmark starts. Quoted and unquoted numeric
selections retain their topology: `0,3` requires at least four allocated PPUs,
and a two-card override is rejected rather than changing the Case layout.
For daily debug runs, `groups.json` retains the command-derived
`cards_per_node` and records the scheduler request separately as
`effective_cards_per_node`. All three launch paths use this recorded request;
the result artifact retains it in `groups.json` and `daily-summary.json`.
This is the requested allocation, not a claim that pending Pods received devices.

The 22:00 Asia/Shanghai schedule selects `full`, which uses the same JSON
generation path as manual dispatch. Monday-Thursday allows 1/2/4/8 nodes and
selects 10,184 Cases in 28 groups with 9 exclusions. Friday allows 1/2/4/8/16
nodes and selects all 10,193 Cases in 29 groups. Manual dispatch keeps the
weekday default unless `allowed_nodes` is explicitly overridden. No legacy
daily CSV dataset remains.

The per-group `cluster-run` queue-and-execution timeout is 720 minutes for the
Monday-Thursday schedule and 2,160 minutes (36 hours) for the Friday schedule.
Manual dispatch uses `execution_timeout_minutes`, which defaults to 720 and can
be set explicitly. The execute Job allows 30 additional minutes for setup and
cleanup. Suite and Case timeouts are unchanged.

Both dispatch choices install HPN SDK 2.1.3 by default. Manual runs can use
`hpn_sdk_url` to select another trusted HTTPS package; dispatch inputs are not
secrets. Credentials, query parameters and fragments are rejected on the hosted
prepare runner before exporting the selected public URL to worker jobs. The
trusted URL is downloaded and installed once per worker
before building or testing, including scheduled runs. Installation failure is
fatal and never falls back to the image package. One run uses one URL.
Signed URLs are not supported by this public dispatch interface. Installer
validation also rejects query parameters and fragments without logging them.
The image and HPN selection are common to M890P and 810E; they are not platform
profile fields. The standard build artifact is shared by both platforms.

Every executed Case inherits `NCCL_IB_ADAPTIVE_ROUTING=1` and
`NCCL_IB_OOO_RQ=1`, including Cases with `NCCL_IB_DISABLE=1`. The executor
forwards both settings through mpirun / `run_test.py` MPI options without
duplicating settings already present in a shared family command.

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
`PCCL_M890P_daily_perf_<run-id>.xlsx` and
`PCCL_810E_daily_perf_<run-id>.xlsx` for dual-platform runs. A manual platform
filter generates only that platform's workbook. Each workbook's overview,
execution totals, performance points, exclusions, Case coverage and metadata
are filtered to its platform; failed or unfinished Cases remain visible.
Ordinary `exit_code` Cases contribute
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
`pccl-daily-performance-report` artifact with 90-day retention, containing both
workbooks when both platforms are selected. Raw logs
remain in the run-owned NAS `results/` tree; the Excel report does not replace
that diagnostic evidence or the existing summary artifact.

XlsxWriter 3.2.5 is pinned with a SHA-256 digest in the existing dependency
artifact and imported directly from its wheel by the summary job; pip is not
required. Report generation is part of the final failure summary, so a missing
required report cannot pass silently.

## Validation boundary

The performance workbook's `Metadata` sheet records the selected image and
effective HPN SDK URL.

Tests compare every generated row with the source-export digests: all 10,193
Cases and all 35 Suite definitions match,
including commands, checks, Suite resources, shared environments and explicit
special Cases. Two legacy `device_api` Plan/Suite identities are explicitly
normalized to the current source names; their 52 commands and resources remain
unchanged. The existing grouping and public input consumer accept the generated
bundle and reproduce the same 10,184 selected Cases, 28 groups and 9 resource
exclusions.

All current daily full Suites are covered, and no generated Python Case files
are produced. Hardware validation is limited to explicitly filtered datasets;
no unfiltered full hardware run has been started.

The 308 `p2pBandwidthLatencyTest` combinations remain outside the reviewed
PCCL daily scope. Three `use_ext_kernel` graph configurations are also skipped
because the public graph binary does not implement that option. Both exclusions
are recorded rather than generated.

Workflow run `43585016370` consumed source-exported families during prepare,
passed both single-node hardware Cases, retained source provenance in the
Case-groups artifact, and completed exact reconciliation and NAS cleanup.
