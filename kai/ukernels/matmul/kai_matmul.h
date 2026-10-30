//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-FileCopyrightText: Copyright 2026 Fujitsu Limited
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kai/ukernels/matmul/kai_matmul_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// For micro-kernel naming and associated packing micro-kernel that goes with it, see:
///   * docs/microkernel_tables.md
///   * kai/ukernels/matmul/README.md
///  This provides information such as the data type and packing format of the buffers.
///  Any information that is not present in the above files is documented here.
///
/// Documentation conventions in this file:
///   * Only required or conditionally required configuration parameters,
///     operands, activation arguments, and flags are documented.
///   * See the *_types.h file for the description of those argument types.
///   * Accumulation data type matches the output data type by default. It is
///     documented along with the API only when the data types differ.
///   * Any argument not listed for a micro-kernel is unused and does not need
///     to be populated.
///

/// Half-precision floating-point matrix multiplication using QMX MOPA instruction.
///
/// Dispatches across 6 tile-size sub-kernels (4vsx4vs, 4vsx8vs, 4vsx16vs,
/// 8vsx4vs, 8vsx8vs, 16vsx4vs) to cover arbitrary MxN output shapes.
///
/// Required CPU features:
///   * FEAT_SME
///   * FEAT_FP16
///
/// Required operands:
///   * lhs - Packed FP16 LHS data.
///   * rhs - Packed FP16 RHS data with packed per-N FP16 accumulator bias.
///   * dst - FP16 output data.
///
/// Optional arguments:
///   * clamp - F32 output clamp values converted to FP16 if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Accumulation: F32, then converted to FP16 output.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_qmx_mopa(void);

/// Half-precision floating-point matrix multiplication using SME2 MOPA instruction.
///
/// Required operands:
///   * lhs, dst
///   * rhs - rhs with per-n accumulator bias
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_sme2_mopa(void);

/// Single-precision floating-point matrix multiplication using SME2 MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///   * FEAT_FP16
///
/// Configuration parameters:
///   * format.bl - Block length. Must be a non-zero multiple of 32.
///
/// Required operands:
///   * lhs - f16p4vsx2 packed FP16 values.
///   * rhs - qsi4c32p16vsx4s1s0sf16 packed with per-block FP16 scale.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * operand.lut.ptr - 16-byte-aligned buffer of 16 32-bit entries mapping packed 4-bit RHS codes to FP16 values.
///   NULL
///     selects the default QSI4 mapping.
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s1s0sf16_f16p_4vsx16vs_sme2_mopa(void);

/// Single-precision floating-point matrix multiplication using QMX MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME
///   * FEAT_FP16
///
/// Configuration parameters:
///   * format.bl - Block length. Must be a non-zero multiple of 32.
///
/// Required operands:
///   * lhs - f16p4vsx2 packed FP16 values.
///   * rhs - qsi4c32p16vsx4s4s0sf16 packed with kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon.
///   * dst - FP32 output matrix.
///
/// S1S0 and S4S0 RHS buffers have matching sizes and strides but are not interchangeable.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa(void);

/// Single-precision floating-point vector-matrix multiplication using SME2 DOT instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Configuration parameters:
///   * format.bl - Block length. Must be a non-zero multiple of 32.
///
/// Required operands:
///   * lhs - qsi8d32p1x4sf16 packed with per-block FP16 scale.
///   * rhs - qsi4c32p16vsx4s1s0sf16 packed with per-block FP16 scale.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * operand.lut.ptr - 16-byte-aligned buffer of 16 32-bit entries mapping packed 4-bit RHS codes to 8-bit values.
///   NULL
///     selects the default QSI4 mapping.
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qsi8d32p1x4sf16_qsi4c32p16vsx4s1s0sf16_i8p_1x16vs_sme2_dot(void);

/// Single-precision floating-point matrix multiplication using SME2 MOPA instruction.
///
/// Required operands:
///   * lhs, dst
///   * rhs - rhs with per-n accumulator bias
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_qmx_mopa(void);

struct kai_matmul_uker_api kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa(void);

/// Matrix multiplication with single-precision floating-point accumulation using SME2 MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Configuration parameters:
///   * format.bl - Block length. Must be a non-zero multiple of 32.
///
/// Required operands:
///   * lhs - qsi8d32p4vsx4sf16 packed with per-block FP16 scale.
///   * rhs - qsi4c32p16vsx4s1s0sf16 packed with per-block FP16 scale.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * operand.lut.ptr - 16-byte-aligned buffer of 16 32-bit entries mapping packed 4-bit RHS codes to 8-bit values.
///   NULL
///     selects the default QSI4 mapping.
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qsi8d32p4vsx4sf16_qsi4c32p16vsx4s1s0sf16_i8p_4vsx16vs_sme2_mopa(void);

/// Matrix multiplication with 32-bit integer accumulation using SME2 MOPA instruction.
///
/// Required operands:
///   * lhs, dst, rhs
///   * bias
///     * acc_bias_m, acc_bias_n
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_sme2_mopa(void);

/// Matrix multiplication with 32-bit integer accumulation using SME MOPA instruction.
///
/// Required operands:
///   * lhs, dst, rhs
///   * bias
///     * acc_bias_m, acc_bias_n
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_i32_u8p4vsx4_u8p4vsx4_i32_i32_8vsx8vs_qmx_mopa(void);

/// Matrix multiplication with 32-bit integer accumulation and FP32 output using SME2 MOPA instruction.
///
/// Required operands:
///   * lhs, dst, rhs
///   * bias
///     * acc_bias_m, acc_bias_n, scale_bias_n
///   * scale
///     * acc_scale_global
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Accumulation: I32, then converted to F32 using global scaling and per column bias
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamping output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_sme2_mopa(void);

/// Matrix multiplication with 32-bit integer accumulation and FP32 output using SME MOPA instruction.
///
/// Required operands:
///   * lhs, dst, rhs
///   * bias
///     * acc_bias_m, acc_bias_n, scale_bias_n
///   * scale
///     * acc_scale_global
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Accumulation: I32, then converted to F32 using global scaling and per column bias
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamping output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_u8p4vsx4_u8p4vsx4_i32_i32_f32_f32_8vsx8vs_qmx_mopa(void);

/// Single-precision floating-point vector-matrix multiplication using SME2 MLA instruction.
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - rhs matrix and per-n accumulator bias vector.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla(void);

/// Half-precision floating-point matrix multiplication using SVE2.1 FDOT instruction.
///
/// Required CPU features:
///   * FEAT_SVE2p1
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N bias.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Accumulation: F32, then converted to F16.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f16_f16_f16p16vsx2bf16_6x16vs_sve2p1_dot(void);

/// Dynamically quantized INT8 vector-matrix multiplication using SVE DOT instructions.
///
/// Required CPU features:
///   * FEAT_SVE with a vector length of exactly 256 bits
///
/// Required operands:
///   * dst
///   * lhs - Packed QAI8DXP data with per-M zero point and scale.
///   * rhs - Packed QSI8CXP data with per-N row sum, scale, and bias.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp8x4_1x8_sve_dot(void);

/// Dynamically quantized INT8 vector-matrix multiplication using SVE DOT instructions.
///
/// Required CPU features:
///   * FEAT_SVE with a vector length of exactly 256 bits
///
/// Required operands:
///   * dst
///   * lhs - Packed QAI8DXP data with per-M zero point and scale.
///   * rhs - Packed QSI8CXP data with per-N row sum, scale, and bias.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp32x4_1x32_sve_dot(void);

/// Dynamically quantized INT8 vector-matrix multiplication using SVE DOT instructions.
///
/// Required CPU features:
///   * FEAT_SVE with a vector length of exactly 256 bits
///
/// Required operands:
///   * dst
///   * lhs - Packed QAI8DXP data with per-M zero point and scale.
///   * rhs - Packed QSI8CXP data with per-N row sum, scale, and bias.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot(void);

/// Half-precision floating-point vector-matrix multiplication using SME2 DOT instruction.
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - rhs matrix and per-n accumulator bias vector.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_sme2_dot(void);

/// Half-precision floating-point vector-matrix multiplication using QMX widening multiply-accumulate instructions.
///
/// Required CPU features:
///   * FEAT_SME
///   * FEAT_FP16
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - rhs matrix and per-n accumulator bias vector.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Accumulation: F32, then converted to F16.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_qmx_dot(void);

/// Statically quantized INT8 matrix multiplication with INT4 RHS using SME2 MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N bias and per-N scale.
///   * dst_bias_global
///
/// Optional arguments:
///   * clamp - INT32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa(void);

/// Quantized 8-bit integer matrix multiplication with 4-bit RHS using QMX MOPA instruction.
///
/// Requires the "s4s0" RHS layout from
/// @ref kai_matmul_pack_rhs_nxk_qsi4cxp8vsx4s4s0sf32bi32_qsi4cx_f32_i32_sme (or its kxn/qsu4cx
/// siblings), whose payload is arranged so each decoded nibble plane is already an SMOPA operand.
/// QMX lacks FEAT_LUT, so the i4 -> i8 expansion is arithmetic; that layout makes it 3 ops per
/// vector with no zip.
///
/// WARNING: NOT interchangeable with the plain
/// @ref kai_matmul_pack_rhs_nxk_qsi4cxp8vsx4sf32bi32_qsi4cx_f32_i32_sme used by the LUTI4-based
/// @ref kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa. Both emit the same
/// packed size, stride and offsets, so a mismatch is not rejected by any assert -- it silently
/// produces wrong results (scrambled K order).
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_qmx_mopa(void);

/// Statically quantized INT8 matrix multiplication using SME2 outer product (MOPA) and SME2.1 quarter tile outer
/// product (MOP4A) instructions.
///
/// Required CPU features:
///   * FEAT_SME2.1
///   * FEAT_SME_MOP4
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N bias and per-N scale.
///   * dst_bias_global
///
/// Optional arguments:
///   * clamp - INT32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4sf32bi32_8vsx8vs_sme2p1_mop4_mopa(void);

/// Single-precision floating-point vector-matrix multiplication using QMX MLA instruction.
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - rhs matrix and per-n accumulator bias vector.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_qmx_mla(void);

/// Statically quantized INT8 matrix-vector multiplication with packed UINT2 RHS using SME2 DOT instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N destination-domain corrected bias and scale.
///   * bias
///     * scale_bias_global - Output zero point as an I32 scalar.
///
/// Optional arguments:
///   * clamp - I32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_sme2_dot(void);

/// Statically quantized INT8 matrix-vector multiplication with packed signed INT2 RHS using QMX DOT instruction.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N destination-domain corrected bias and scale.
///   * bias
///     * scale_bias_global - Output zero point as an I32 scalar.
///
/// Optional arguments:
///   * clamp - I32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8_qsi2cxp16vsx4s12s8s4s0sf32bi32_1x64vs_qmx_dot(void);

/// Statically quantized INT8 matrix multiplication with packed UINT2 RHS using SME2 MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N destination-domain corrected bias and scale.
///   * bias
///     * scale_bias_global - Output zero point as an I32 scalar.
///
/// Optional arguments:
///   * clamp - I32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8p8vsx4_qsu2cxp16vsx4sf32bi32_8vsx16vs_sme2_mopa(void);

/// Statically quantized INT8 matrix multiplication with packed signed INT2 RHS using QMX MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N destination-domain corrected bias and scale.
///   * bias
///     * scale_bias_global - Output zero point as an I32 scalar.
///
/// Optional arguments:
///   * clamp - I32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8p8vsx4_qsi2cxp16vsx4s12s8s4s0sf32bi32_8vsx16vs_qmx_mopa(void);

/// Statically quantized INT8 matrix multiplication using SME2 MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N bias and per-N scale.
///   * dst_bias_global
///
/// Optional arguments:
///   * clamp - INT32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_sme2_mopa(void);

/// Statically quantized INT8 GEMM using SME MOPA instruction.
///
/// Dispatches across 6 tile-size sub-kernels (4vsx4vs, 4vsx8vs, 4vsx16vs,
/// 8vsx4vs, 8vsx8vs, 16vsx4vs) to cover arbitrary M×N output shapes.
///
/// Required operands:
///   * lhs, rhs, dst
///   * bias
///     * scale_bias_global
///
/// Optional arguments:
///   * clamp - I8 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamping output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8p4vsx4_qsi8cxp4vsx4bi32sf32_8vsx8vs_qmx_mopa(void);

/// Statically quantized INT8 vector-matrix multiplication using SME2 DOT instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N bias and per-N scale.
///   * dst_bias_global
///
/// Optional arguments:
///   * clamp - INT32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_sme2_dot(void);

/// Statically quantized INT8 vector-matrix multiplication using QMX DOT instruction.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - RHS matrix with per-N bias and per-N scale.
///   * dst_bias_global
///
/// Optional arguments:
///   * clamp - INT32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8_qsi8cxp4vsx4bi32sf32_1x32vs_qmx_dot(void);

/// Statically quantized INT8 vector-matrix multiplication with packed INT4 RHS using SME2 DOT instruction.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - Packed RHS matrix with per-N bias and per-N scale.
///   * bias
///     * scale_bias_global - Output zero point as an I32 scalar.
///
/// Optional arguments:
///   * clamp - I32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot(void);

/// Statically quantized INT8 vector-matrix multiplication with packed INT4 RHS using SME1 (QMX)
/// SDOT instructions.
///
/// QMX counterpart of @ref kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot. FEAT_LUT
/// is unavailable, so the INT4 -> INT8 expansion is arithmetic rather than a LUTI4 table lookup.
///
/// The RHS must be packed by one of:
///   * @ref kai_matmul_pack_rhs_nxk_qsi4cxp8vsx4s4s0sf32bi32_qsi4cx_f32_i32_sme
///   * @ref kai_matmul_pack_rhs_nxk_qsi4cxp8vsx4s4s0sf32bi32_qsu4cx_f32_i32_sme
///   * @ref kai_matmul_pack_rhs_kxn_qsi4cxp8vsx4s4s0sf32bi32_qsi4cx_f32_i32_sme
///   * @ref kai_matmul_pack_rhs_kxn_qsi4cxp8vsx4s4s0sf32bi32_qsu4cx_f32_i32_sme
/// i.e. the same "s4s0" packed buffer consumed by the
/// @ref kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_qmx_mopa GEMM kernel, so a
/// single packed RHS can serve both the GEMM and the GEMV path.
///
/// The plain "sf32bi32" packers used by the sme2_dot sibling are NOT interchangeable with these:
/// every stride, offset and packed size is identical and only the arrangement of nibbles within the
/// payload differs, so a mismatch is not rejected by any assert -- it silently produces wrong
/// results.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Required operands:
///   * dst
///   * lhs
///   * rhs - Packed RHS matrix with per-N bias and per-N scale.
///   * bias
///     * scale_bias_global - Output zero point as an I32 scalar.
///
/// Optional arguments:
///   * clamp - I32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4s4s0sf32bi32_1x8vs_qmx_dot(void);

/// Matrix multiplication with FP16 packed LHS and QAI4C32P RHS with FP32 output using SME2 MOPA.
///
/// Required CPU features:
///   * FEAT_SME2
///   * FEAT_FP16
///
/// Configuration parameters:
///   * format.bl - Block length. Must be 32.
///
/// Required operands:
///   * lhs - FP16 data packed in 4vsx2 panels.
///   * rhs - qai4c32p16vsx4s1s0sf16 packed with per-block FP16 offset and scale.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * clamp - FP32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa(void);

/// Matrix multiplication with FP16 packed LHS and signed-domain QAI4C32P RHS with FP32 output using QMX MOPA.
///
/// Required CPU features:
///   * FEAT_SME
///   * FEAT_FP16
///
/// Configuration parameters:
///   * format.bl - Block length. Must be 32.
///
/// Required operands:
///   * lhs - FP16 data packed in 4vsx2 panels.
///   * rhs - qai4c32p16vsx4s4s0sf16 packed with signed s4s0 payloads and adjusted per-block FP16 offset and scale.
///   * dst - FP32 output matrix.
///
/// S1S0 and S4S0 RHS buffers have matching sizes and strides but are not interchangeable.
///
/// Optional arguments:
///   * clamp - FP32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s4s0sf16_4vsx16vs_qmx_mopa(void);

/// Vector-matrix multiplication with dynamically quantized INT8 packed LHS and QAI4C32P RHS packed inputs with FP32
/// output using SME2 DOT.
///
/// Required CPU features:
///   * FEAT_SME2
///
/// Configuration parameters:
///   * format.bl - Block length. Must be 32.
///
/// Required operands:
///   * lhs - qsi8d32p1x4sf16 data packed with per-block FP16 sum and scale.
///   * rhs - qai4c32p16vsx4s1s0sf16 packed with per-block FP16 offset and scale.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * clamp - FP32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot(void);

/// Vector-matrix multiplication with dynamically quantized INT8 packed LHS and signed-domain QAI4C32P RHS packed
/// inputs with FP32 output using QMX DOT.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Configuration parameters:
///   * format.bl - Block length. Must be 32.
///
/// Required operands:
///   * lhs - qsi8d32p1x4sf16 data packed with per-block FP16 sum and scale.
///   * rhs - qai4c32p16vsx4s4s0sf16 packed with signed s4s0 payloads and adjusted per-block FP16 offset and scale.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * clamp - FP32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s4s0sf16_1x16vs_qmx_dot(void);

/// Matrix multiplication with single-precision floating-point accumulation using QMX DOT instruction.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Configuration parameters:
///   * format.bl - Block length. Must be a non-zero multiple of 32.
///
/// Required operands:
///   * lhs - qsi8d32p1x4sf16 packed with per-block FP16 scale.
///   * rhs - qsi4c32p16vsx4s4s0sf16 packed with kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qsi8d32p1x4sf16_qsi4c32p16vsx4s4s0sf16_i8p_1x16vs_qmx_dot(void);

/// Matrix multiplication with single-precision floating-point accumulation using QMX MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Configuration parameters:
///   * format.bl - Block length. Must be a non-zero multiple of 32.
///
/// Required operands:
///   * lhs - qsi8d32p4vsx4sf16 packed with per-block FP16 scale.
///   * rhs - qsi4c32p16vsx4s4s0sf16 packed with kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qsi8d32p4vsx4sf16_qsi4c32p16vsx4s4s0sf16_i8p_4vsx16vs_qmx_mopa(void);

/// Single-precision floating-point matrix multiplication with statically-blockwise quantized unsigned INT4 RHS,
/// using SME MOPA instruction.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Configuration parameters:
///   * bl - The block length, in number of K values per block. Must be a multiple of 32.
///
/// Required operands:
///   * dst
///   * lhs - Dynamically quantized asymmetric per-row (qai8dxp) LHS, packed with @ref
///     kai_lhs_quant_pack_qai8dxp_f32.
///   * rhs - Packed RHS matrix, obtained with @ref kai_matmul_pack_rhs_kxn_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme or
///   @ref
///     kai_matmul_pack_rhs_nxk_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qai8dxp4vsx4_qsi4c32p16vsx4_4vsx16vs_sme_mopa(void);

/// Single-precision floating-point vector-matrix multiplication with statically-blockwise quantized unsigned INT4
/// RHS, using SME DOT instruction.
///
/// Required CPU features:
///   * FEAT_SME
///
/// Configuration parameters:
///   * bl - The block length, in number of K values per block. Must be a multiple of 32.
///
/// Required operands:
///   * dst
///   * lhs - Dynamically quantized asymmetric per-row (qai8dxp) LHS, packed with @ref
///     kai_lhs_quant_pack_qai8dxp_f32.
///   * rhs - Packed RHS matrix, obtained with @ref kai_matmul_pack_rhs_kxn_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme or
///   @ref
///     kai_matmul_pack_rhs_nxk_qsi4c32p16vsx4s4s0_qsu4c32_f32_bf16_sme.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p16vsx4_1x16vs_sme_dot(void);

/// Matrix-matrix multiplication with dynamically quantized INT8 packed LHS and INT8 packed RHS packed inputs with FP32
/// output using SVE MMLA.
///
/// Required CPU features:
///   * FEAT_SVE
///   * FEAT_I8MM
///
/// Required operands:
///   * lhs - qai8dxp4x8 data packed with per-row I32 zeropoint, and FP32 scale.
///   * rhs - qsi8cxp4vsx8 packed with per-column I32 sum,  F32 bias and FP32 scale.
///   * dst - FP32 output matrix.
///
/// Optional arguments:
///   * clamp - FP32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm(void);

/// Dynamically quantized matrix multiplication with blockwise INT4 RHS using SVE I8MM instructions.
///
/// Required CPU features:
///   * FEAT_SVE with a vector length of exactly 256 bits
///   * SVE FEAT_I8MM instructions
///
/// The scalable name reserves support for other vector lengths. This implementation requires vscale = 2,
/// with LHS panels of 4 rows and an output block of 16x8.
///
/// Required configuration parameters:
///   * format.bl - RHS quantization block length. It must be a non-zero multiple of 32, and K must be a non-zero
///     multiple of it.
///
/// Required operands:
///   * dst
///   * lhs - Packed QAI8DXP data with per-M zero point and scale.
///   * rhs - Packed QSI4C32P data with BF16 per-block scales, F32 per-N row sums, and F32 per-N bias.
///
/// Optional arguments:
///   * clamp - F32 output clamp values if KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP flag is set.
///
/// Supported flags:
///   * KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP - Clamp output data.
///
/// @return The micro-kernel API.
struct kai_matmul_uker_api kai_matmul_clamp_f32_qai8dxp2vsx8sf32_qsi4c32p8x8s16s0sbf16bf32_8vsx8_sve_i8mm(void);

#ifdef __cplusplus
}  // extern "C"
#endif
