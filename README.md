# tiny-cuda-kernels

A minimal CUDA kernel playground optimized for a remote Jetson workflow.

## What You Get

- CMake-based CUDA project structure
- Sample vector-add kernel with correctness validation
- Micro-benchmark timing using CUDA events
- Scripts to sync, build, and run on Jetson over SSH

## Repository Layout

```
.
├── CMakeLists.txt
├── include/
│   └── tck/
│       ├── cuda_utils.cuh
│       └── vector_add.cuh
├── scripts/
│   ├── build_on_jetson.sh
│   ├── open_latest_ncu_report.sh
│   ├── profile_on_jetson_ncu.sh
│   ├── run_on_jetson.sh
│   └── sync_to_jetson.sh
└── src/
    ├── vector_add/
    │   └── vector_add.cu
    └── vector_add_main.cu
```

## Prerequisites

- Jetson with CUDA toolkit installed
- SSH alias configured (current default in scripts: `jetson-n1`)
- Local machine with `rsync` and `ssh`

## Quick Start (Jetson Remote Loop)

From the repo root:

```bash
./scripts/sync_to_jetson.sh
./scripts/build_on_jetson.sh
./scripts/run_on_jetson.sh vector_add
```

That runs `vector_add` with defaults:

- `N = 16,777,216` elements
- `iterations = 100`
- `warmup = 10`

Or run any built target directly:

```bash
./scripts/run_on_jetson.sh <target> [args ...]
```

## Detailed Benchmark Output

The benchmark now reports:

- Correctness status
- Kernel latency distribution: `avg`, `min`, `p50`, `p95`, `p99`, `max`, `stddev`
- Kernel effective memory throughput (GB/s)
- Host-to-device and device-to-host transfer time and bandwidth
- Per-iteration kernel timing samples (non-CSV mode)

For CSV-friendly summary output:

```bash
./scripts/run_on_jetson.sh vector_add 16777216 200 20 --csv
```

CSV columns:

`n,iterations,warmup,avg_ms,min_ms,p50_ms,p95_ms,p99_ms,max_ms,stddev_ms,kernel_gbps,h2d_ms,h2d_gbps,d2h_ms,d2h_gbps`

## Build Directly On Jetson

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/vector_add 16777216 100 10
```

## CUDA Architecture

For best Jetson performance, pass your target SM architecture at configure time:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
```

Adjust the value for your Jetson model.

## Nsight Compute (Detailed Kernel Profiling)

Generate an Nsight Compute report for any target on Jetson and copy it back locally:

```bash
./scripts/profile_on_jetson_ncu.sh vector_add 16777216 50 10
```

Generic profiling arguments:

- `<target>` executable name under `build/`
- `[app args ...]` arguments passed to the target

Optional report tag:

```bash
REPORT_TAG=my_run ./scripts/profile_on_jetson_ncu.sh vector_add 16777216 50 10
```

Environment options:

- `JETSON_HOST` (default: `jetson-n1`)
- `JETSON_DIR` (default: `~/dev/tiny-cuda-kernels`)
- `CUDA_HOME` (default: `/usr/local/cuda`)
- `NCU_SET` (default: `launchstats`; can set `full` for deeper metrics)
- `USE_SUDO` (`auto`, `always`, `interactive`, `never`; default: `auto`)

Output report path:

- Local: `reports/ncu/<report_tag>_<timestamp>.ncu-rep`
- Remote: `~/dev/tiny-cuda-kernels/reports/ncu/<report_tag>_<timestamp>.ncu-rep`

Open the newest local report in one command:

```bash
./scripts/open_latest_ncu_report.sh
```

Optional:

- Different report folder: `./scripts/open_latest_ncu_report.sh reports/ncu`
- Custom app name: `NCU_APP_NAME="NVIDIA Nsight Compute" ./scripts/open_latest_ncu_report.sh`
- Dry run: `DRY_RUN=1 ./scripts/open_latest_ncu_report.sh`

If Jetson blocks profiling due to performance counter permissions, run with sudo:

```bash
USE_SUDO=always ./scripts/profile_on_jetson_ncu.sh vector_add 16777216 50 10
```

If non-interactive sudo is unavailable, run with interactive sudo prompt:

```bash
USE_SUDO=interactive ./scripts/profile_on_jetson_ncu.sh vector_add 16777216 50 10
```