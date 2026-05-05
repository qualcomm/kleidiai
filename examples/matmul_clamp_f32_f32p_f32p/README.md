<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# matmul_clamp_f32_f32p_f32p Example

Matrix multiplication of two single-precision floating-point (FP32) matrices with an optional
bias vector and a clamp operation on the output. Both LHS and RHS are packed before the matmul.

## Overview

This example tests one micro-kernel variant:

| Test | Micro-kernel | LHS format | RHS format | Threading | Iterations |
|------|-------------|------------|------------|-----------|-----------|
| TEST[0] | `kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa` | Packed FP32 (`f32p2vlx1`, SME) | Packed FP32+bias (`f32p2vlx1biasf32`, SME) | Multi-threaded | 16 |

### Packing functions used

| Role | Function |
|------|----------|
| LHS pack | `kai_lhs_pack_f32p2vlx1_f32_sme` |
| RHS pack | `kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme` |

### Multi-threading

Each thread independently packs its slice of the LHS matrix and then runs the matmul on that slice.
Work is distributed across threads by dividing the M dimension into equal-sized chunks aligned to `m_step`.

### Performance measurement

The QMX MOPA kernel is run **16 iterations** to obtain a more accurate average performance measurement.
LHS packing and RHS packing are done once before the timing loop.

## Hardware requirements

| Variant | Required features |
|---------|------------------|
| QMX MOPA | AArch64 + `FEAT_SVE2` + Qualcomm QMX |

## Building

From the `examples/matmul_clamp_f32_f32p_f32p` directory:

### Linux®-target (cross-compile)

```bash
mkdir -p build && cd build
cmake \
  -DCMAKE_C_COMPILER=/path/to/aarch64-none-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=/path/to/aarch64-none-linux-gnu-g++ \
  -DCMAKE_BUILD_TYPE=Release \
  ../
make -j$(nproc)
```

### Android™-target

```bash
mkdir -p build && cd build
cmake \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=30 \
  -DCMAKE_BUILD_TYPE=Release \
  ../
make -j$(nproc)
```

## Usage

```
./matmul_clamp_f32_f32p_f32p [--threads <count>]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--threads <N>` | Number of worker threads | `1` |
| `--threads=<N>` | Alternative syntax | `1` |
| `-t <N>` | Short form | `1` |
| `--help` / `-h` | Print usage and exit | — |

### Examples

```bash
# Single-threaded (default)
./matmul_clamp_f32_f32p_f32p

# 4 threads
./matmul_clamp_f32_f32p_f32p --threads 4

# Alternative syntax
./matmul_clamp_f32_f32p_f32p --threads=4
```

### Expected output

```
Using 4 thread(s) for computations.
Matrix dimensions: M=512 N=512 K=512

Testing matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa
TEST[0] = PASSED
- ukernel: matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa
- Iterations: 16
- Total Performance time: <time> us
- Avg Performance time per iteration: <time> us
```

## Matrix dimensions

The example uses fixed dimensions `M=512`, `N=512`, `K=512`.

- Both M and N are multiples of common tile sizes, giving full-tile execution.
- The reference implementation uses scalar FP32 accumulation with bias.
- Correctness is verified with a 0.1% relative tolerance.
