# PCCL Inspector Plugin

The PCCL Inspector is a plugin for the PPU Collective Communication Library (PCCL) that provides detailed, per-communicator, per-collective performance and metadata logging. It is designed to help users analyze and debug PCCL collective operations by generating structured JSON output for each operation.

## Related Documentation

- **[Performance Exporter](exporter/example/README.md)** - Tool for analyzing and visualizing PCCL performance data from inspector logs

## Folder Location

The Inspector plugin source is located in:

```
ext-profiler/inspector/
```

## Building the Inspector Plugin

To build the Inspector plugin, run:

```bash
make
```

Build PCCL first to generate `build/include/`, and set `PPU_SDK` to the installed SDK directory.

### Build Options

The Makefile supports the following build option:

- **DEBUG=1**: Enable debug build with additional debugging information

Example debug build:
```bash
make DEBUG=1
```

### Build Output

The build process creates:
- `libpccl-profiler-inspector.so`: The main inspector plugin library
- `version.cc`: Auto-generated version information from git

## Using PCCL Inspector

### Key Differences from Normal PCCL Usage

The main difference between running PCCL with the Inspector plugin versus running PCCL normally is the addition of environment variables that enable detailed performance logging:

**Normal PCCL Run:**
```bash
# Standard PCCL execution
./your_pccl_application
```

**PCCL Inspector Run:**
```bash
# PCCL Inspector enabled execution
export PCCL_PROFILER_PLUGIN=/path/to/pccl/ext-profiler/inspector/libpccl-profiler-inspector.so
export PCCL_INSPECTOR_ENABLE=1
export PCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
./your_pccl_application
```

### Required Environment Variables

- `PCCL_PROFILER_PLUGIN=/path/to/pccl/ext-profiler/inspector/libpccl-profiler-inspector.so`
  Loads the Inspector plugin into PCCL.
- `PCCL_INSPECTOR_ENABLE=1`
  Enables the Inspector plugin.
- `PCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=<interval>`
  Sets the interval (in microseconds) for the internal dump thread to write output. Example: `500`.
- `PCCL_INSPECTOR_DUMP_DIR=<output_dir>` (optional)
  Sets the output directory for logs. If not set, defaults to `pccl-inspector-unknown-jobid` or `pccl-inspector-<slurm_job_id>` if running under SLURM.
- `PCCL_INSPECTOR_DUMP_VERBOSE=<0|1>` (optional)
  Enables verbose output including event trace information. Set to `1` to enable, `0` to disable (default).

### Example Usage

**Single Node:**
```bash
export PCCL_PROFILER_PLUGIN=/path/to/pccl/ext-profiler/inspector/libpccl-profiler-inspector.so
export PCCL_INSPECTOR_ENABLE=1
export PCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
/path/to/pccl-tests/build/all_reduce_perf -b 8 -e 16G -f 2 -g 8
```

**Multi-Node (SLURM):**
```bash
# Add these environment variables to your SLURM script
export PCCL_PROFILER_PLUGIN=/path/to/pccl/ext-profiler/inspector/libpccl-profiler-inspector.so
export PCCL_INSPECTOR_ENABLE=1
export PCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
export PCCL_INSPECTOR_DUMP_DIR=/path/to/logs/${SLURM_JOB_ID}/

# Then run your normal PCCL application
srun your_pccl_application
```

## Output Example

Each output file contains JSON objects with the following structure:

```json
{
  "header": {
    "id": "0x690fbf08e16dfb",
    "rank": 0,
    "n_ranks": 2,
    "nnodes": 1
  },
  "metadata": {
    "inspector_output_format_version": "v4.0",
    "git_rev": "",
    "rec_mechanism": "pccl_profiler_interface",
    "dump_timestamp_us": 1792413371108940,
    "hostname": "example-host",
    "pid": 1787173
  },
  "coll_perf": {
    "coll": "AllReduce",
    "coll_sn": 33,
    "coll_msg_size_bytes": 131072,
    "coll_exec_time_us": 11,
    "coll_timing_source": "kernel_gpu",
    "coll_algobw_gbs": 11.92,
    "coll_busbw_gbs": 11.92
  }
}
```

## Output Example Verbose

To enable verbose output with event trace information, set the `PCCL_INSPECTOR_DUMP_VERBOSE=1` environment variable:

```bash
export PCCL_INSPECTOR_DUMP_VERBOSE=1
```

This will include additional event trace information in the JSON output, showing the sequence of callbacks and timestamps for each individual event.

```json
{
  "header": {
    "id": "0x6fbf0eb2f9753f",
    "rank": 7,
    "n_ranks": 8,
    "nnodes": 1
  },
  "metadata": {
    "inspector_output_format_version": "v4.0",
    "git_rev": "",
    "rec_mechanism": "pccl_profiler_interface",
    "dump_timestamp_us": 1792413683928413,
    "hostname": "example-host",
    "pid": 1880913
  },
  "coll_perf": {
    "coll": "AllReduce",
    "coll_sn": 0,
    "coll_msg_size_bytes": 2147483648,
    "coll_exec_time_us": 15221,
    "coll_timing_source": "kernel_gpu",
    "coll_algobw_gbs": 141.08,
    "coll_busbw_gbs": 246.89,
    "event_trace_sn": {
      "coll_start_sn": 1,
      "coll_stop_sn": 2,
      "kernel_events": [
        {
          "channel_id": 0,
          "kernel_start_sn": 10,
          "kernel_stop_sn": 36,
          "kernel_record_sn": 35
        },
        {
          "channel_id": 1,
          "kernel_start_sn": 12,
          "kernel_stop_sn": 38,
          "kernel_record_sn": 37
        },
        {
          "channel_id": 2,
          "kernel_start_sn": 9,
          "kernel_stop_sn": 40,
          "kernel_record_sn": 39
        },
        {
          "channel_id": 3,
          "kernel_start_sn": 3,
          "kernel_stop_sn": 42,
          "kernel_record_sn": 41
        }
      ]
    },
    "event_trace_ts": {
      "coll_start_ts": 1792413683908206,
      "coll_stop_ts": 1792413683908220,
      "kernel_events": [
        {
          "channel_id": 0,
          "kernel_start_ts": 1792413683925708,
          "kernel_stop_ts": 1792413683925770,
          "kernel_record_ts": 1792413683925770
        },
        {
          "channel_id": 1,
          "kernel_start_ts": 1792413683925711,
          "kernel_stop_ts": 1792413683925770,
          "kernel_record_ts": 1792413683925770
        },
        {
          "channel_id": 2,
          "kernel_start_ts": 1792413683925654,
          "kernel_stop_ts": 1792413683925770,
          "kernel_record_ts": 1792413683925770
        },
        {
          "channel_id": 3,
          "kernel_start_ts": 1792413683910454,
          "kernel_stop_ts": 1792413683925770,
          "kernel_record_ts": 1792413683925770
        }
      ]
    }
  }
}
```

Multiple such JSON objects are written, one per collective operation per communicator.

## Output Directory

- By default, output files are written to:
  - `pccl-inspector-unknown-jobid` (if no SLURM job ID is present)
  - `pccl-inspector-<slurm_job_id>` (if running under SLURM)
- You can override this with the `PCCL_INSPECTOR_DUMP_DIR` environment variable.

## Additional Notes

- The plugin is compatible with standard PCCL workflows and can be used in both single-node and multi-node (SLURM) environments.
- For more details, see the source code and comments in `ext-profiler/inspector/`.

