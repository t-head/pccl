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

## Workflow datasets

The experimental branch adds two explicit `workflow_dispatch` choices:

- `generated-smoke`: two Cases in one execution group.
- `generated-full`: 3,994 generated Cases. The current 1/2/4-node resource
  policy selects 3,985 Cases in ten groups and records nine eight-node Cases as
  excluded.

The existing `smoke` and `full` datasets and the 22:00 Asia/Shanghai schedule
are unchanged. The schedule still selects the legacy `full` dataset; no
production cutover is implied by this experiment.

Each generated bundle contains `generation.json` with definition and Suite
hashes, the selected profile, and the exact Case-key hash. The workflow copies
it into the Case-groups artifact. `export.json` retains the existing consumer
schema and records `xtest_version=not-used` explicitly.

## Validation boundary

Migration tests compare every generated row with the existing export: all 54
op_perf Cases and all 3,940 postsil Cases match exactly, including commands,
checks, Suite resources and the two explicit special Cases. The existing
grouping and public input consumer accept the generated bundles.

Only these two families are covered. Other Suites still use their existing
exports, and no generated Python Case files are produced. Hardware validation
is limited to the explicitly triggered generated smoke dataset.

Workflow run `43447932105` generated two smoke Cases during prepare, passed both
single-node hardware Cases, retained the two-family generation manifest in the
Case-groups artifact, and completed exact reconciliation and NAS cleanup. No
generated-full hardware run was started.
