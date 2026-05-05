<!--
    SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

    SPDX-License-Identifier: Apache-2.0
-->

# matmul_clamp_f32_bf16p_bf16p Example

Matrix multiplication of two brain floating-point (BF16) matrices with FP32 output,
optional bias, and a clamp operation on the result.

## Overview

This example tests three micro-kernel variants and validates each against a scalar
reference implementation:

| Test | Micro-kernel | LHS format | RHS format | Threading |
|------|-------------|------------|------------|-----------|
| TEST[0] | `kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot` | Packed BF16 (`bf16p1x4`, NEON) | Packed BF16+bias (`bf16p12x4b`, NEON) | Single-threaded |
| TEST[1] | `kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla` | Packed BF16 (`bf16p8x4`, NEON) | Packed BF16+bias (`bf16p12x4b`, NEON) | Single-threaded |
| TEST[2] | `kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa` | Packed BF16 (`bf16p2vlx2`, SME) | Packed BF16+bias (`bf16p2vlx2b`, SME) | Multi-threaded |

### Packing functions used

| Role | Function |
|------|----------|
| TEST[0] LHS pack | `kai_lhs_quant_pack_bf16p1x4_f32_neon` |
| TEST[1] LHS pack | `kai_lhs_quant_pack_bf16p8x4_f32_neon` |
| TEST[0]/[1] RHS pack | `kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon` |
| TEST[2] LHS pack | `kai_lhs_pack_bf16p2vlx2_f32_sme` |
| TEST[2] RHS pack | `kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme` |

### Multi-threading (TEST[2])

The QMX MOPA kernel supports multi-threaded execution. Each thread independently packs
its slice of the LHS matrix and then runs the matmul on that slice. Work is distributed
across threads by dividing the M dimension into equal-sized chunks aligned to `m_step`.

### Performance measurement (TEST[2])

The QMX MOPA kernel is run **16 iterations** to obtain a more accurate average
performance measurement. LHS packing is done once before the timing loop.

## Hardware requirements

| Variant | Required features |
|---------|------------------|
| TEST[0] NEON dot | AArch64 + `FEAT_BF16` |
| TEST[1] NEON mmla | AArch64 + `FEAT_BF16` |
| TEST[2] QMX MOPA | AArch64 + `FEAT_SVE2` + `FEAT_BF16` + Qualcomm QMX |

## Building

From the `examples/matmul_clamp_f32_bf16p_bf16p` directory:

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
./matmul_clamp_f32_bf16p_bf16p [--threads <count>]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--threads <N>` | Number of worker threads for the QMX MOPA kernel (TEST[2]) | `1` |
| `--threads=<N>` | Alternative syntax | `1` |
| `-t <N>` | Short form | `1` |
| `--help` / `-h` | Print usage and exit | — |

### Examples

```bash
# Single-threaded (default)
./matmul_clamp_f32_bf16p_bf16p

# 4 threads for the QMX MOPA kernel
./matmul_clamp_f32_bf16p_bf16p --threads 4

# Alternative syntax
./matmul_clamp_f32_bf16p_bf16p --threads=4

# Short form
./matmul_clamp_f32_bf16p_bf16p -t 4
```

### Expected output

```
Using 4 thread(s) for computations.
Matrix dimensions: M=512 N=512 K=512

TEST[matmul_clamp_f32_bf16p_bf16p]
- ukernel: matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot
- Status: PASSED
- Performance: <time> ns

TEST[matmul_clamp_f32_bf16p_bf16p]
- ukernel: matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla
- Status: PASSED
- Performance: <time> ns

Testing matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa
TEST[2] = PASSED
- ukernel: matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa
- Iterations: 16
- Total time: <time> us
- Avg time per iteration: <time> us
```

## Matrix dimensions

The example uses fixed dimensions `M=512`, `N=512`, `K=512`.

- `K=512` satisfies the QMX MOPA requirement that K is a multiple of `kr=2`.
- Both M and N are multiples of common tile sizes, giving full-tile execution.
- The NEON kernels handle arbitrary M, N, K.
