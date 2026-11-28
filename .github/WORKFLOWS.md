# GitHub workflows

CI and module review run independently. Pull requests targeting `main` must pass both
`PCCL Tests Gate` and `PCCL Review Gate`.

- [CI checks and PPU tests](docs/ci.md) — maintained by the `ci` module: test gates,
  PPU tests, daily runs, CSV Cases, build artifacts and NAS cache.
- [Contribution and review workflows](docs/review.md) — maintained by the `workflow`
  module: reviewer assignment, module approval and commit message previews.
