<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# matmul_clamp_f16_f16_f16p Example

Matrix multiplication of two half-precision floating-point (FP16) matrices with FP16 output,
optional bias, and a clamp operation on the result.

## Overview

This example tests two micro-kernel variants:

| Test | Micro-kernel | LHS format | RHS format | Threading |
|------|-------------|------------|------------|-----------|
| NEON | `kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla` | Unpacked FP16 | Packed FP16+bias (NEON) | Single-threaded |
| QMX MOPA | `kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa` | Packed FP16 (`x16p2vlx2`, SME) | Packed FP16+bias (`x16p2vlx2b`, SME) | Multi-threaded |

### Packing functions used

| Role | Function |
|------|----------|
| NEON RHS pack | `kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon` |
| QMX MOPA LHS pack | `kai_lhs_pack_x16p2vlx2_x16_sme` |
| QMX MOPA RHS pack | `kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme` |

### Multi-threading (QMX MOPA)

Each thread independently packs its slice of the LHS matrix and then runs the matmul on that slice.
Work is distributed across threads by dividing the M dimension into equal-sized chunks aligned to `m_step`.

### Performance measurement (QMX MOPA)

The QMX MOPA kernel is run **16 iterations** to obtain a more accurate average performance measurement.
LHS packing is done once before the timing loop.

## Hardware requirements

| Variant | Required features |
|---------|------------------|
| NEON MLA | AArch64 + `FEAT_FP16` |
| QMX MOPA | AArch64 + `FEAT_SVE2` + `FEAT_FP16` + Qualcomm QMX |

## Building

From the `examples/matmul_clamp_f16_f16_f16p` directory:

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
./matmul_clamp_f16_f16_f16p [--threads <count>]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--threads <N>` | Number of worker threads for the QMX MOPA kernel | `1` |
| `--threads=<N>` | Alternative syntax | `1` |
| `-t <N>` | Short form | `1` |
| `--help` / `-h` | Print usage and exit | — |

### Examples

```bash
# Single-threaded (default)
./matmul_clamp_f16_f16_f16p

# 4 threads for the QMX MOPA kernel
./matmul_clamp_f16_f16_f16p --threads 4

# Alternative syntax
./matmul_clamp_f16_f16_f16p --threads=4
```

### Expected output

```
Using 4 thread(s) for computations.
TEST[matmul_clamp_f16_f16_f16p]
- ukernel: matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla
- Status: PASSED

TEST QMX MOPA [M=512 N=512 K=512]
Testing matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa
TEST[matmul_clamp_f16_f16p_f16p]
- ukernel: matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa
- Status: PASSED
- Iterations: 16
- Total Performance time: <time> us
- Avg Performance time per iteration: <time> us
