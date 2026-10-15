# Topology tests

Build and run these tests only inside a remote PPU container, after building the worktree library and setting up the PPU SDK. Put the worktree `build/lib` first in `LD_LIBRARY_PATH` and verify library resolution and hashes before executing tests.

From the repository root (with Meson and Ninja installed):

```bash
bash tests/setup_3rd.sh
meson setup tests/build tests -Dppu_archs=ppu_10,ppu_15
ninja -C tests/build pxnPeerSelectionTest
meson test -C tests/build pxnPeerSelection --print-errorlogs
```

For incremental test builds, use `ninja -C tests/build pxnPeerSelectionTest`. The test is also registered as `pxnPeerSelection` with Meson. Its optional first argument overrides the topology XML path.

The PXN test requires default routing settings with level 2 enabled; clear inherited NCCL/PCCL tuning before running it. It loads the checked-in topology, constructs contiguous and interleaved two-node rank mappings, and calls the actual library selector. The matrix covers every permutation of four devices and two eight-device layouts, checks candidate eligibility, singleton selection, group balance, call-order independence and fallback behavior, and asserts the reordered `[0,2,1,3]` counterexample. This is topology validation, not a live communication or performance test.
