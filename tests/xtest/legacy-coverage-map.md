# Legacy GitHub catalog coverage map

`legacy-coverage-map.json` is a pinned migration diagnostic for the former
96-entry GitHub catalog. It is not an input to Case selection, manifest
generation, execution, or the result gate. The executable sanity selection
continues to come only from `source-catalog.json`, which is exported from the
reviewed `pccl_test_suites` Plan, Suite, and `github_sanity` label rule.

For source revision `bbbeb207dd7f28980b55336660da0b47866e47aa`, the map records:

- 6 exact, supported M890P entries;
- 6 near, supported M890P entries, mapping the former two process/thread
  layouts onto 3 source Cases with explicit devices `0,1`;
- 60 exact entries whose source Suite resource is not the selected M890P
  single-node resource; and
- 24 entries with no source command match.

The source-side mapper records configuration load errors for diagnosis but
does not treat invalid configurations as source matches. The production
exporter remains fail-closed and will not publish a partial catalog.
