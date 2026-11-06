# GIN device API checks

Build on Linux after sourcing the SDK environment and building PCCL:

```sh
make -C tests/device_api PCCL_HOME="$PWD/build"
export LD_LIBRARY_PATH="$PWD/build/lib:$LD_LIBRARY_PATH"
export PCCL_GIN_PLUGIN=none PCCL_GIN_TYPE=3 PCCL_MNNVL_ENABLE=0
mpirun -np 2 tests/device_api/gin_window 0
mpirun -np 2 tests/device_api/gin_config
mpirun -np 2 tests/device_api/gin_wait
```

`gin_window 0` verifies device-backed GIN-only windows; `gin_window 1`
verifies VMM HOST_NUMA windows. Each mode checks every allocated context,
put/get payloads, registration aliases, peer LSA rejection and invalid flags.
An optional second argument requests custom stride: with four ranks,
`gin_window 1 2` checks stride two. The stride must divide the LSA team size
and half the world size so the selected transfer peer belongs to the subset.
Custom stride retains dense provider connections; sparse allocation is not
implemented. Unsupported strides are rejected rather than silently remapped.
Use an even number of ranks and assign one rank to each available device.
For multiple hosts, place the same executable and library at the same path.
Select the network interfaces and GIN connection count for the environment.
Tests must run on idle devices; use a process timeout to detect a hung peer.

`gin_config` checks capability reporting, backend selection, collective request
agreement, legacy output canaries and LSA-only creation after GIN activation.
`gin_wait` checks immediate success, timeout, invalid bit widths and delayed
indexed/VA signals and counters, including cooperative status broadcasts.

These checks require MPI and a functioning GIN backend and are run manually;
they are not part of the default build or the hosted CI jobs.
