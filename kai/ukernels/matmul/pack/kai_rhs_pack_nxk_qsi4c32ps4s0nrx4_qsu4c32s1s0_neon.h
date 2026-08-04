//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>

#include "kai/kai_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Get the n step value.
/// The micro-kernel can process any N values. However, the starting N index to
/// be processed must be a multiple of n step.
///
/// @param[in] nr The number of columns written by the matmul micro-kernel
///
/// @return the n step value
size_t kai_get_n_step_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(size_t nr);

/// Gets the offset in bytes for the RHS matrix (not packed), which holds
/// the int4 values in a N x K matrix, where N is number of rows and K is the number of columns.
///
/// Two int4 values are stored in one byte.
///        The lower order part of the byte (low) holds the first nibble (K-index + 0).
///        The higher order of the byte holds the second nibble (K-index + 1).
///
/// @param[in] n_idx      Row index in the RHS matrix (not packed).
/// @param[in] rhs_stride The number of bytes in in each row of the RHS matrix (not packed)
///
/// @return the offset in bytes to the RHS matrix (not packed)
size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(
    size_t n_idx,  //
    size_t rhs_stride);

/// Get the row stride in bytes to the packed RHS matrix
///
/// @param[in] k        The number of columns in the RHS matrix (not packed).
/// @param[in] nr       The number of columns written by the matmul micro-kernel.
/// @param[in] kr       The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr       The number of kr splits. It can be 1 (no splits) up to kr.
/// @param[in] bl       The block length, which defines the number of K values stored in a single block. It must be a
/// multiple of 32.
/// @param[in] scale_dt Block scale data type
///
/// @return the stride in bytes to the packed RHS matrix
size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(
    size_t k,   //
    size_t nr,  //
    size_t kr,  //
    size_t sr,  //
    size_t bl,  //
    enum kai_datatype scale_dt);

/// Gets the offset in bytes for the packed RHS matrix.
///
/// @param[in] n_idx    Row index in the RHS matrix (not packed).
/// @param[in] k        The number of columns in the RHS matrix (not packed).
/// @param[in] nr       The number of columns written by the matmul micro-kernel
/// @param[in] kr       The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr       The number of kr splits. It can be 1 (no splits) up to kr.
/// @param[in] bl       The block length, which defines the number of K values stored in a single block. It must be a
/// multiple of 32.
/// @param[in] scale_dt Block scale data type
///
/// @return the offset in bytes to the packed RHS matrix
size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(
    size_t n_idx,  //
    size_t k,      //
    size_t nr,     //
    size_t kr,     //
    size_t sr,     //
    size_t bl,     //
    enum kai_datatype scale_dt);

/// Gets the size in bytes for the quantized and packed RHS matrix.
///
/// @param[in] n  The number of rows in the RHS matrix (not packed)
/// @param[in] k  The number of columns in the RHS matrix (not packed).
/// @param[in] nr The number of columns written by the matmul micro-kernel
/// @param[in] kr The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] sr The number of kr splits. It can be 1 (no splits) up to kr.
/// @param[in] bl The block length, which defines the number of K values stored in a single block. It must be a multiple
/// of 32.
/// @param[in] scale_dt Block scale data type
///
/// @return the packed RHS matrix size in bytes
// clang-format off
size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(
    size_t n,   //
    size_t k,   //
    size_t nr,  //
    size_t kr,  //
    size_t sr,  //
    size_t bl,  //
    enum kai_datatype scale_dt);
// clang-format on

/// Runs the RHS packing micro-kernel.
///
/// The RAW (unpacked) input convention is the same as
/// kai_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon: the int4 values are stored in a N x K
/// matrix, where N is number of rows and K is the number of columns. Two int4 values are stored
/// in one byte. The lower order part of the byte (low) holds the first nibble (K-index + 0). The
/// higher order of the byte holds the second nibble (K-index + 1).
///
/// The PACKED output layout differs from kai_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon only
/// in how the int4 weight nibbles are grouped within a block. Two int4 K values are stored in one
/// byte, grouped into 8-K-value packets (4 bytes each). Packet p covers K[8p..8p+7], relative to
/// the start of each bl-sized block (i.e. packet numbering resets every bl K-values): byte b
/// (0..3) of the packet holds K-index 8p+b in the low nibble and K-index 8p+b+4 in the high
/// nibble. Values are signed two's complement (-8..7). For example, if the block length is 32,
/// the values within one column's block are stored in the following byte order:
///   byte(s4, s0),byte(s5, s1),byte(s6, s2),byte(s7, s3),
///   byte(s12, s8),byte(s13, s9),byte(s14, s10),byte(s15, s11),
///   byte(s20, s16),byte(s21, s17),byte(s22, s18),byte(s23, s19),
///   byte(s28, s24),byte(s29, s25),byte(s30, s26),byte(s31, s27)
/// Everything else (bf16 block scale, per-column reduction sum, per-column bias, nr-column
/// cross-interleave) is identical to kai_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon.
///
/// @param[in]  num_groups   The number of groups. It must be 1. Currently unused.
/// @param[in]  n            The number of rows in the RHS matrix (not packed).
/// @param[in]  k            The number of columns in the RHS matrix (not packed).
/// @param[in]  nr           The number of columns written by the matmul micro-kernel. It must be a multiple of 4.
/// @param[in]  kr           The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in]  sr           The number of kr splits. It can be 1 (no splits) up to kr.
///                          However, kr must be multiple of sr.
/// @param[in]  bl           The block length, which defines the number of
///                          K values stored in a single block. It must be a multiple of 32.
/// @param[in]  rhs          The RHS matrix containing the 4-bit values.
///                          Size in bytes is expected to be greater than or equal to n * k * (sizeof(uint8_t) / 2).
/// @param[in]  rhs_stride   The number of bytes per row in bytes of the RHS matrix
/// @param[in]  bias         The biases.
/// @param[in]  scale        The per-block quantization scales.
///                          The scale data type must be provided with the params object.
///                          Supported scale data types are FP32, FP16 and BF16.
/// @param[in]  scale_stride The number of bytes per row in bytes of the scale matrix
/// @param[out] rhs_packed   The packed RHS matrix.
/// @param[in]  extra_bytes  Extra bytes to append to the end of each row of the packed RHS matrix. Currently unused.
/// @param[in]  params       Parameters for the micro-kernel.
void kai_run_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(
    size_t num_groups,    //
    size_t n,             //
    size_t k,             //
    size_t nr,            //
    size_t kr,            //
    size_t sr,            //
    size_t bl,            //
    const uint8_t* rhs,   //
    size_t rhs_stride,    //
    const float* bias,    //
    const void* scale,    //
    size_t scale_stride,  //
    void* rhs_packed,     //
    size_t extra_bytes,   //
    const struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params* params);

#ifdef __cplusplus
}
#endif
