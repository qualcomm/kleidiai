<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
    SPDX-License-Identifier: Apache-2.0
-->

# matmul_clamp_qai8_qai8p_qsi8cxp Example

Matrix multiplication with packed int8 asymmetric LHS (qai8p) and int8 symmetric per-channel
RHS with scale+bias (qsi8cxpsb), producing int8 output with requantization and clamp.

## Overview

| Test | Micro-kernel | LHS | RHS | Output | Threading | Iterations |
|------|-------------|-----|-----|--------|-----------|-----------|
| TEST[0] | `kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa` | Packed qai8p (SME) | Packed qsi8cxpsb (SME) | int8 | Multi-threaded | 10000 |

**Fixed shape:** M=512, N=512, K=512

### Packing functions

| Role | Function |
|------|----------|
| LHS pack | `kai_lhs_pack_x8p2vlx4_x8_sme` |
| RHS pack | `kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme` |

### Requantization

The output is requantized to int8 using `kai_matmul_requantize32_params`:
- `min_value = -128`, `max_value = 127`, `output_zero_point = 0`
- `scale_multiplier = lhs_scale × rhs_scale / output_scale`

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
./matmul_clamp_qai8_qai8p_qsi8cxp [--threads <N>]
```

### Examples

```bash
./matmul_clamp_qai8_qai8p_qsi8cxp --threads 4
```

### Expected output

```
Using 4 thread(s) for computations.
Matrix dimensions: M=512 N=512 K=512

Testing matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa
TEST[0] = PASSED
- ukernel: matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa
- Iterations: 10000
- Total Performance time: <time> us
- Avg Performance time per iteration: <time> us
