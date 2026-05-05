<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
    SPDX-License-Identifier: Apache-2.0
-->

# matmul_clamp_f32_qai8dxp_qsi8cxp Example

Matrix multiplication with dynamically quantized 8-bit asymmetric per-row LHS (qai8dxp)
and 8-bit symmetric per-channel RHS (qsi8cxp), producing FP32 output with a clamp operation.

## Overview

| Test | Micro-kernel | LHS | RHS | Threading | Iterations |
|------|-------------|-----|-----|-----------|-----------|
| TEST[0] | `kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa` | Packed qai8dxp (SME) | Packed qsi8cxp (NEON) | Multi-threaded | 10000 |

**Fixed shape:** M=512, N=512, K=512

### Packing functions

| Role | Function |
|------|----------|
| LHS pack | `kai_lhs_quant_pack_qai8dxp_f32` |
| RHS pack | `kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon` |

## Hardware requirements

| Required features |
|------------------|
| AArch64 + `FEAT_SVE2` + Qualcomm QMX |

## Building

### Android™-target

```bash
mkdir -p build && cd build
cmake \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=30 \
  -DCMAKE_BUILD_TYPE=Release ../
make -j$(nproc)
```

## Usage

```
./matmul_clamp_f32_qai8dxp_qsi8cxp [--threads <N>]
```

### Examples

```bash
./matmul_clamp_f32_qai8dxp_qsi8cxp --threads 4
```

### Expected output

```
Using 4 thread(s) for computations.
Matrix dimensions: M=512 N=512 K=512

Testing matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa
TEST[0] = PASSED
- ukernel: matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa
- Iterations: 10000
- Total Performance time: <time> us
- Avg Performance time per iteration: <time> us
