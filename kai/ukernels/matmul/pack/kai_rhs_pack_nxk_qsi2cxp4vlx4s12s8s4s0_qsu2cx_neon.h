//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef kai_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon_params
#define kai_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon_params kai_rhs_pack_qs4cxs1s0_param
#endif

/// Get the n step value.
///
/// @param[in] nr The number of columns written by the matmul micro-kernel.
///
/// @return The n step value.
size_t kai_get_n_step_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(size_t nr);

/// Gets the offset in bytes for the unpacked RHS matrix.
///
/// @param[in] n_idx      Row index in the RHS matrix. It must be a multiple of n_step.
/// @param[in] rhs_stride The number of bytes in each row of the RHS matrix.
///
/// @return The offset in bytes to the RHS matrix.
size_t kai_get_rhs_offset_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(size_t n_idx, size_t rhs_stride);

/// Gets the row stride in bytes for the packed RHS matrix.
///
/// @param[in] k  The common dimension between the LHS and RHS matrices.
/// @param[in] nr The number of N columns in a packed panel. It must be divisible by 4.
/// @param[in] kr The number of K values loaded by the innermost matmul loop.
/// @param[in] sr The number of kr splits.
///
/// @return The stride in bytes to the packed RHS matrix.
size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(
    size_t k, size_t nr, size_t kr, size_t sr);

/// Gets the offset in bytes for the packed RHS matrix.
///
/// @param[in] n_idx Row index in the unpacked RHS matrix. It must be a multiple of n_step.
/// @param[in] k     The common dimension between the LHS and RHS matrices.
/// @param[in] nr    The number of N columns in a packed panel.
/// @param[in] kr    The number of K values loaded by the innermost matmul loop.
/// @param[in] sr    The number of kr splits.
///
/// @return The offset in bytes to the packed RHS matrix.
size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr);

/// Gets the size in bytes for the packed RHS matrix.
///
/// @param[in] n  The number of rows in the unpacked RHS matrix.
/// @param[in] k  The common dimension between the LHS and RHS matrices.
/// @param[in] nr The number of N columns in a packed panel.
/// @param[in] kr The number of K values loaded by the innermost matmul loop.
/// @param[in] sr The number of kr splits.
///
/// @return The packed RHS matrix size in bytes.
size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(
    size_t n, size_t k, size_t nr, size_t kr, size_t sr);

/// Packs an NxK QSU2 matrix into a signed 2-bit, K-regrouped layout.
///
/// The input stores four unsigned 2-bit values per byte, with K+0 in bits [1:0]
/// through K+3 in bits [7:6]. Each source code q is recoded to (q - 2) & 3.
/// For one RHS row and each 16-K unit beginning at K, the output is:
///
///   byte 0 = [K+12 | K+8  | K+4  | K+0]
///   byte 1 = [K+13 | K+9  | K+5  | K+1]
///   byte 2 = [K+14 | K+10 | K+6  | K+2]
///   byte 3 = [K+15 | K+11 | K+7  | K+3]
///
/// Each bracket lists signed 2-bit fields from most significant to least significant.
/// Rows are stored consecutively within each 16-K unit, so each vector-length region
/// supplies one N quarter to the corresponding QMX MOPA accumulator tile. This private
/// format is not compatible with the generic QSU2 packers or their consumers.
///
/// @param[in]  num_groups  The number of groups. It must be 1.
/// @param[in]  n           The number of rows in the RHS matrix.
/// @param[in]  k           The number of columns in the RHS matrix. It must be a multiple of 32.
/// @param[in]  nr          The number of N columns in a packed panel. It must be divisible by 4.
/// @param[in]  kr          The number of K values loaded by the innermost matmul loop.
/// @param[in]  sr          The number of kr splits.
/// @param[in]  rhs         The RHS matrix containing unsigned 2-bit values.
/// @param[in]  bias        Per-row biases, or NULL to use zero biases.
/// @param[in]  scale       Per-row scale factors.
/// @param[out] rhs_packed  The packed RHS matrix.
/// @param[in]  extra_bytes Extra bytes appended to each packed panel. It must be 0.
/// @param[in]  params      Packing parameters. rhs_zero_point must be 2.
/// @param[in]  lut_arg     Unused; signed recoding and row-sum semantics are fixed.
void kai_run_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const float* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon_params* params, const int32_t* lut_arg);

#ifdef __cplusplus
}
#endif
