# On-demand public test lists

This experimental path generates Case inputs in the public `pccl` repository at
workflow prepare time. It uses only the Python standard library and the existing
public CSV consumer; xtest and mmengine are neither installed nor imported.

The fourteen Runner-backed daily definitions are published by the full source
JSON exporter at source revision `1f7d18a5058c93982b387cb34f2b1f80bcb92c6c`.
`test_lists/source-export.json` records their exact 8,899-Case, 34-Suite
real-Runner comparison and SHA-256 digests. CSV is not an input or output of
that publication step. The 35 native supplement Cases and the separate
two-Case source-native adapter are not included in that source-export total.

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
- `test_lists/native_unit_test_shm.json`: direct `collPcclUnitTest` and
  `hggcGraphUnitTest` commands for the public native unit-test experiment. This
  is a new native test family, not a translation of the legacy 1,640 Cases.
- `test_lists/source_native_unit_test_shm.json`: two bare native Cases exported
  directly from their source Python configurations. Its `provenance` records
  the source revision, exact source Case paths and command adaptations. This
  is a feasibility sample, not yet a generated replacement for the full native
  parameter matrix.
- `test_lists/suites.json`: Plan/Suite identity, nodes, cards, timeout and shared
  environment references.
- `test_lists/envs/`: shared public runtime environments.
- `test_lists/source-export.json`: full source selection, per-family row
  digests, definition hashes and the combined Case/Suite equivalence manifest.

Generated CSV is a temporary execution artifact, not a maintained input. The
legacy `daily-*-csv` directories are used only by migration tests as equivalence
oracles and by the unchanged legacy datasets.

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

The two-family example is a subset. The workflow's `generated-full` selection
passes all fourteen daily definitions to the same command. The source-native
sample has a separate validation dataset and is not included in that selection.

## Workflow datasets

The experimental branch adds two explicit `workflow_dispatch` choices:

- `generated-smoke`: two Cases in one execution group.
- `generated-full`: all 8,934 daily Cases in 35 Suites, including 35 native
  unit Cases. The current 1/2/4-node resource policy selects 8,215 Cases in
  21 groups and records 710
  eight-node and 9 sixteen-node Cases as excluded.
- `generated-native-smoke`: one direct AllReduce Case and one direct HGGC
  simple_test Case in a single execution group. The native family currently
  generates 27 Cases in its full profile but is not yet included in
  `generated-full`.
- `generated-native-full`: all 35 direct native Cases. This is a validation
  dataset and does not run the legacy 1,640 Cases.
- `generated-native-extra-full`: eight additional public native Cases covering
  send/recv, communicator shrink, memory management and group simulation.
  Run `43576246963` passed all eight and completed NAS cleanup, so they are now
  included alongside the original 27 native Cases in `generated-full`.
- `generated-source-native-smoke`: two source-derived AllReduce and AllGather
  Cases. They execute `collPcclUnitTest` directly with four devices, FC/LL and
  ten iterations; neither the legacy `unit_test.py` wrapper nor the source-only
  `--auto-detect` option is used. This dataset is limited to a single-node
  feasibility run and does not imply full source-family coverage.

The existing `smoke` and `full` datasets and the 22:00 Asia/Shanghai schedule
are unchanged. The schedule still selects the legacy `full` dataset; no
production cutover is implied by this experiment.

Each generated bundle contains `generation.json` with definition and Suite
hashes, the selected profile, and the exact Case-key hash. The workflow copies
it into the Case-groups artifact. `export.json` retains the existing consumer
schema and records `xtest_version=not-used` explicitly. When a definition has
source provenance, the generation manifest retains it separately from the
public `pccl` revision.

## Validation boundary

Migration tests compare every generated row with the existing export and with
the source-export digests: all 8,899 Cases and all 34 Suite definitions match,
including commands, checks, Suite resources, shared environments and explicit
special Cases. Two legacy `device_api` Plan/Suite identities are explicitly
normalized to the current source names; their 52 commands and resources remain
unchanged. The existing grouping and public input consumer accept the generated
bundle and reproduce the same 8,180 selected Cases, twenty groups and 719
resource exclusions.

All fourteen Runner-backed daily full Suites are covered, and no generated
Python Case files are produced. The legacy CSV remains as a secondary migration
oracle and as the unchanged input of the legacy datasets; it is not read by the
public generator. Hardware validation is limited to the explicitly triggered
smoke and native datasets listed below, not the complete generated-full set.

Workflow run `43447932105` generated two smoke Cases during prepare, passed both
single-node hardware Cases, retained the two-family generation manifest in the
Case-groups artifact, and completed exact reconciliation and NAS cleanup. No
generated-full hardware run was started.

The standard build now also packages `collPcclUnitTest` and
`hggcGraphUnitTest`. Their commands execute the binaries directly; no
`unit_test.py` launcher is used. Native validation is limited to
`generated-native-smoke` until those commands pass on hardware.

Workflow run `43567684114` passed all 27 native Cases and completed exact
reconciliation and NAS cleanup. These 27 Cases are now included in
`generated-full`; the legacy 1,640 Cases remain outside this experiment.
