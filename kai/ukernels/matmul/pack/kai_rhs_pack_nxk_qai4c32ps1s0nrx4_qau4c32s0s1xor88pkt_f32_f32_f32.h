//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//
#pragma once

#include <stddef.h>

#include "kai/kai_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef kai_rhs_pack_nxk_qai4c32p_params
#define kai_rhs_pack_nxk_qai4c32p_params kai_rhs_pack_qs4cxs1s0_param
#endif

/// xor88pkt RHS packer for the zip-free QMX micro-kernels
///
/// Packs a qai4c32 NxK RHS matrix for @ref kai_run_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_qmx_dot,
/// @ref kai_run_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_qmx_dot and their SMOPA siblings
/// @ref kai_run_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_qmx_mopa and
/// @ref kai_run_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_qmx_mopa.
///
/// Use this "s0s1" variant when source column c holds k=2c in its HIGH nibble and k=2c+1 in its low nibble.
///
/// Three things distinguish it from @ref kai_run_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon,
/// and together they let the micro-kernels decode a nibble with a single shift or a single
/// immediate AND instead of an lsr + zip1/zip2 + sub sequence:
///  -# every packed byte is XOR-ed with 0x88, which turns the two zero-point-8 nibbles into
///     signed two's complement, removing the runtime "sub #8";
///  -# K is grouped into 8-value packets. Output byte b of a packet holds K[8p+b] in its low
///     nibble and K[8p+b+4] in its high nibble, so a 32-bit slot spans four K-adjacent values
///     and feeds the 4-way reduction of SDOT/SMOPA directly;
///  -# the per-block scale is pre-divided by 16, compensating the micro-kernels' top-justified
///     nibble read (which scales every value by 16).
///
/// The packed size, row stride and per-block scale/zero-point/bias offsets are byte-identical to
/// the plain packer above.
///
/// WARNING: precisely because the layout is size- and offset-compatible, feeding this packer's
/// output to any other qai4c32p micro-kernel -- or feeding the plain packer's output to the qmx
/// kernels listed above -- is not rejected by any assert and silently produces wrong results.


/// Gets the offset in bytes for the RHS matrix (not packed), which holds
/// the int4 values in a N x K matrix, where N is number of rows and K is the number of columns.
/// Two int4 K values are stored in one byte. These values are stored in blocks
///
/// @param[in] n_idx      Row index in the RHS matrix (not packed).
/// @param[in] rhs_stride The number of bytes in in each row of the RHS matrix (not packed)
///
/// @return the offset in bytes to the RHS matrix (not packed)
size_t kai_get_rhs_offset_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1xor88pkt_f32_f32_f32(
    size_t n_idx,        //
    size_t rhs_stride);  //

/// Gets the offset in bytes for the packed RHS matrix.
///
/// @param[in] n_idx    Row index in the RHS matrix (not packed).
/// @param[in] k        The common dimension between the LHS and RHS matrix (K)
/// @param[in] nr       The number of columns written by the matmul micro-kernel
/// @param[in] kr       The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] bl       The block length, which defines the number of K values stored in a single block. It must be a
/// multiple of 32.
///
/// @return the offset in bytes to the packed RHS matrix
size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1xor88pkt_f32_f32_f32(
    size_t n_idx,  //
    size_t k,      //
    size_t nr,     //
    size_t kr,     //
    size_t bl      //
);

/// Gets the size in bytes for the quantized and packed RHS matrix.
///
/// @param[in] n  The number of rows in the RHS matrix (not packed)
/// @param[in] k  The number of columns in the RHS matrix (not packed).
/// @param[in] nr The number of columns written by the matmul micro-kernel
/// @param[in] kr The number of columns loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in] bl The block length, which defines the number of K values stored in a single block. It must be a multiple
/// of 32.
///
/// @return the packed RHS matrix size in bytes
size_t kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1xor88pkt_f32_f32_f32(
    size_t n,   //
    size_t k,   //
    size_t nr,  //
    size_t kr,  //
    size_t bl   //
);

/// Run the micro-kernel to pack the RHS matrix.
///
/// @note   The int4 values are stored in a N x K matrix, where N is number of rows and K is the number of columns.
///         Two int4 values are stored in one byte.
///         nrx4 => this function can take in generic nr values but the input is expected to have a block depth of 4
///         Block depth is calculated as kr / sr. The values of these parameters are defined in the matmul ukernel.
///
/// SHARED packet layout, usable by BOTH the SDOT and the SMOPA micro-kernels of this family.
///
/// Within each block, K is grouped into "packets" of 8 consecutive values. Packet p (covering
/// K[8p .. 8p+7]) is stored as one 4-byte group per N row, at 32-bit slot (p * nr + n):
///   byte b of that slot holds  low nibble = K[8p + b],  high nibble = K[8p + b + 4].
/// Every packed byte is XOR 0x88'd, converting the zero-point-8 nibble values to signed
/// two's-complement, so the micro-kernels decode with a plain top-justified shift/mask
/// (lsl #4 for the low nibble, AND 0xF0 for the high nibble) -- no ZIP1/ZIP2 reassembly and
/// no runtime zero-point subtraction. The decode scales every value by 16 (top-justified byte
/// read as signed); this packer pre-divides the block scale by 16 to compensate.
///
/// Both micro-kernels read register r (r = 0..3) as "packet p, N rows r*(nr/4) .. r*(nr/4)+(nr/4)-1":
///   - the SDOT kernel addresses it with sequential `#r, MUL VL` offsets;
///   - the SMOPA kernel addresses it with strided `base + r * cntw` offsets;
/// these compute identical byte addresses, which is what makes one layout serve both. Each
/// instruction's own 4-way byte reduction then consumes the 4 consecutive K values that a
/// decoded nibble stream provides.
///
/// The packed size and the per-block zero-point / scale / bias offsets are byte-for-byte
/// identical to @ref kai_run_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s1s0_f32_f32_f32_neon:
/// per block, nr * (bl / 2) data bytes, then nr zero-point floats, then nr scale floats;
/// the nr bias floats follow all blocks.
///
/// @param[in]  num_groups  The number of groups. It must be 1.
/// @param[in]  n           The number of columns of the output matrix (N).
/// @param[in]  k           The common dimension between the LHS and RHS matrix (K).
/// @param[in]  nr          The number of N rows to interleave on the same output row.
/// @param[in]  kr          The number of K values loaded in the single inner most loop of the matmul micro-kernel.
/// @param[in]  sr          The number of kr splits. It can be 1 (no splits) up to kr.
///                         However, kr must be multiple of sr.
/// @param[in]  bl          The block length, which defines the number of
///                         K values stored in a single block. It must be a multiple of 32.
/// @param[in]  rhs         The RHS matrix containing the 4-bit values.
///                         Size in bytes is expected to be greater than or equal to n * k * (sizeof(uint8_t) / 2).
/// @param[in]  zero        The zero point.
/// @param[in]  bias        The biases.
/// @param[in]  scale       The scale for each output channel.
/// @param[out] rhs_packed  The packed RHS matrix.
/// @param[in]  extra_bytes Extra bytes to append to the end of each row of the packed RHS matrix.
/// @param[in]  params      Parameters for the micro-kernel.
void kai_run_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1xor88pkt_f32_f32_f32(
    size_t num_groups,   //
    size_t n,            //
    size_t k,            //
    size_t nr,           //
    size_t kr,           //
    size_t sr,           //
    size_t bl,           //
    const uint8_t* rhs,  //
    const void* zero,    //
    const void* bias,    //
    const void* scale,   //
    void* rhs_packed,    //
    size_t extra_bytes,  //
    const struct kai_rhs_pack_nxk_qai4c32p_params* params);
#ifdef __cplusplus
}
#endif
