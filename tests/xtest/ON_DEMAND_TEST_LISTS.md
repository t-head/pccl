# On-demand public test lists

This experimental path generates Case inputs in the public `pccl` repository at
workflow prepare time. It uses only the Python standard library and the existing
public CSV consumer; xtest and mmengine are neither installed nor imported.

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
- `test_lists/suites.json`: Plan/Suite identity, nodes, cards, timeout and shared
  environment references.
- `test_lists/envs/`: shared public runtime environments.

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
passes all fourteen definitions to the same command.

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

The existing `smoke` and `full` datasets and the 22:00 Asia/Shanghai schedule
are unchanged. The schedule still selects the legacy `full` dataset; no
production cutover is implied by this experiment.

Each generated bundle contains `generation.json` with definition and Suite
hashes, the selected profile, and the exact Case-key hash. The workflow copies
it into the Case-groups artifact. `export.json` retains the existing consumer
schema and records `xtest_version=not-used` explicitly.

## Validation boundary

Migration tests compare every generated row with the existing export: all 8,899
Cases and all 34 Suite definitions match exactly, including commands, checks,
Suite resources, shared environments and explicit special Cases. The existing
grouping and public input consumer accept the generated bundle and reproduce
the same 8,180 selected Cases, twenty groups and 719 resource exclusions.

All current daily full Suites are covered, and no generated Python Case files
are produced. The legacy CSV remains as a migration oracle and as the unchanged
input of the legacy datasets. Hardware validation is limited to the explicitly
triggered generated smoke dataset.

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
