<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# matmul_clamp_f32_qai8dxp_qsi4cxp Example

Matrix multiplication with dynamically quantized 8-bit asymmetric per-row LHS (qai8dxp)
and 4-bit symmetric per-channel LHS (qsi4cxp), producing FP32 output with a clamp operation.

## Overview

This example tests multiple micro-kernel variants across several matrix shapes and RHS formats,
then runs the QMX MOPA kernel with multi-threading and 16 iterations for accurate performance measurement.

### NEON variants (existing tests — unchanged)

Each NEON variant is tested across 6 matrix shapes with both NxK and KxN RHS formats:

| Variant | Architecture |
|---------|-------------|
| `matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod` | NEON dotprod |
| `matmul_clamp_f32_qai8dxp1x8_qsi4cxp8x8_1x8x32_neon_dotprod` | NEON dotprod |
| `matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_4x4x32_neon_i8mm` | NEON i8mm |
| `matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x8_8x4x32_neon_i8mm` | NEON i8mm |
| `matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_4x8x32_neon_i8mm` | NEON i8mm |
| `matmul_clamp_f32_qai8dxp4x8_qsi4cxp8x8_8x8x32_neon_i8mm` | NEON i8mm |
| `matmul_clamp_f32_qai8dxp4x4_qsi4cxp8x4_8x8x32_neon_dotprod` | NEON dotprod |
| `matmul_clamp_f32_qai8dxp4x8_qsi4cxp4x4_16x4x32_neon_dotprod` | NEON dotprod |
| `matmul_clamp_f32_qai8dxp1x4_qsi4cxp4x4_1x4_neon_dotprod` | NEON dotprod |

### QMX MOPA variant (new — multi-threaded, 16 iterations)

| Variant | Architecture | Threading | Iterations |
|---------|-------------|-----------|-----------|
| `matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa` | QMX MOPA | Multi-threaded | 16 |

**Fixed shape:** M=512, N=512, K=512, RHS format=NxK

### Packing functions used for QMX MOPA

| Role | Function |
|------|----------|
| LHS pack | `kai_lhs_quant_pack_qai8dxp_f32` (same as NEON) |
| RHS pack | `kai_rhs_pack_nxk_qsi4cxps1s0_qsu4cxs1s0_neon` |

### Multi-threading (QMX MOPA)

Each thread independently packs its slice of the LHS matrix and then runs the matmul on that slice.
Work is distributed across threads by dividing the M dimension into equal-sized chunks aligned to `m_step`.

### Performance measurement (QMX MOPA)

The QMX MOPA kernel is run **16 iterations** to obtain a more accurate average performance measurement.
LHS packing is done once before the timing loop.

## Hardware requirements

| Variant | Required features |
|---------|------------------|
| NEON dotprod | AArch64 + `FEAT_DOTPROD` |
| NEON i8mm | AArch64 + `FEAT_I8MM` |
| QMX MOPA | AArch64 + `FEAT_SVE2` + Qualcomm QMX |

## Building

From the `examples/matmul_clamp_f32_qai8dxp_qsi4cxp` directory:

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
./matmul_clamp_f32_qai8dxp_qsi4cxp [--threads <count>]
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
./matmul_clamp_f32_qai8dxp_qsi4cxp

# 4 threads for the QMX MOPA kernel
./matmul_clamp_f32_qai8dxp_qsi4cxp --threads 4

# Alternative syntax
./matmul_clamp_f32_qai8dxp_qsi4cxp --threads=4
```

### Expected output (excerpt)

```
Using 4 thread(s) for computations.
------------

TEST[1, 33,32]
Testing RHS format = N x K
Testing matmul_clamp_f32_qai8dxp1x8_qsi4cxp4x8_1x4x32_neon_dotprod
TEST[0] = PASSED
...

TEST QMX MOPA [512, 512, 512]
Testing RHS format = N x K
Testing matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa
TEST[0] = PASSED
- Iterations: 16
- Total Performance time: <time> us
- Avg Performance time per iteration: <time> us
