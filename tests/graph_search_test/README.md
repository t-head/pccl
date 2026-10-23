# Graph Search Test

## Overview

Unit tests for validating PCCL's topology-aware communication graph generation. Tests verify that PCCL correctly discovers hardware topology from XML descriptions and computes optimal ring and tree patterns for collective operations across different PPU configurations (TP1/2/4/8), node counts (1-2 nodes), and interconnect topologies (OAM, PIX, PXN). Generated graphs are compared against reference files for correctness.

## Build

```shell
source /usr/local/PPU_SDK/envsetup.sh     # or your SDK path

# Option 1: Compile pccl and all tests including graph_search_test
cd pccl && ./build.sh --use_sdk --enable_tests

# Option 2: Use pccl release artifact to build graph_search_test standalone
source your_path_to_pccl/envsetup.sh
cd pccl/tests/graph_search_test && make -j
```

## Usage

### Running Tests

```shell
# Run all tests
cd tests/graph_search_test
./build/graphSearchTest

# Run with verbose PCCL debug output
./build/graphSearchTest --verbose

# Enable PXN (cross-NIC communication) tests
./build/graphSearchTest --pxn

# Specify custom topology directory
./build/graphSearchTest --root /path/to/topo_and_reference_graph/

# Run specific test cases using Google Test filters
./build/graphSearchTest --gtest_filter=graphSearchTestWithoutPXN.*
./build/graphSearchTest --gtest_filter=*8PPU4CX7*
```

### Command-Line Arguments

- `-r, --root <path>` Root directory containing topology XML files and reference graphs (default: auto-detected relative to binary location)
- `-v, --verbose` Enable verbose PCCL debug output
- `-p, --pxn` Enable PXN (cross-NIC communication) mode. When not specified, main sets `PCCL_PXN_DISABLE=1` to disable PXN. Does not select test cases (use `--gtest_filter`).
- Standard Google Test flags: `--gtest_filter`, `--gtest_list_tests`, etc.

## Examples

### Run Specific Configuration Tests

```shell
# Run all WithoutPXN tests
./build/graphSearchTest --gtest_filter="graphSearchTestWithoutPXN*"

# Run all WithPXN tests
./build/graphSearchTest --pxn --gtest_filter="graphSearchTestWithPXN*"

# Test only 8-PPU with CX7 NICs
./build/graphSearchTest --gtest_filter=*8PPU4CX7*
```

### Debug Mode

```shell
# Enable verbose PCCL logging
./build/graphSearchTest --verbose --gtest_filter=*8PPU4CX7Tp1*
```
