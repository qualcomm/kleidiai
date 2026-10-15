//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// nrx4 => this function can take in generic nr values but the input is expected to have a block depth of 4.
// Block depth is calculated as kr / sr. The values of these parameters are defined in the matmul ukernel.

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_rhs_pack_kxn_qsi4c32ps4s0nrx4_qsu4c32s1s0.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_sum_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);
static const size_t kai_nr_multiple_of = 4;
static const size_t kai_bl_multiple_of = 32;
static const size_t kai_rhs_zero_point = 8;

// The lsl/and decode used by the consuming matmul kernel leaves both nibbles in the upper 4
// bits of the byte (unshifted), which is numerically 16x too large. This is compensated by
// pre-dividing the stored per-block scale by 1/16 at pack time (mirroring
// kai_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon's kai_pre_scaled_rhs_scale_factor).
// The reduction-sum (rsum) computation below must use the ORIGINAL, undivided scale -- rsum
// reflects the true dequantized weight sum for the LHS zero-point correction and has nothing
// to do with how the kernel later decodes packed nibbles.
static const float kai_pre_scaled_rhs_scale_factor = 1.0F / 16.0F;

static size_t kai_get_num_blocks_per_row(const size_t k, const size_t bl) {
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    return kai_roundup(k, bl) / bl;
}

static size_t kai_get_num_bytes_per_block(const size_t bl, const size_t num_bytes_multiplier_rhs) {
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    return (bl / 2) + num_bytes_multiplier_rhs;
}

static size_t kai_get_rhs_packed_offset_end_of_all_blocks(
    // clang-format off
    const size_t k,
    const size_t nr,
    const size_t kr,
    const size_t bl,
    const size_t num_bytes_multiplier_rhs) {
    // clang-format on
    KAI_ASSERT((bl % kr) == 0);
    KAI_ASSERT((nr % kai_nr_multiple_of) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);

    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block(bl, num_bytes_multiplier_rhs);

    return (nr * num_bytes_per_block * num_blocks_per_row);
}

// Reads the signed (zero-point-subtracted) nibble for row k_idx, column n_idx of a K x N
// matrix packed 2 int4 values per byte along the N axis (byte holds N_idx and N_idx+1).
static inline int32_t kai_read_signed_nibble_kxn(
    const uint8_t* rhs, const size_t rhs_stride, const size_t k_idx, const size_t n_idx) {
    const uint8_t byte = rhs[k_idx * rhs_stride + (n_idx / 2)];
    const uint8_t nibble = ((n_idx % 2) == 0) ? (byte & 0x0F) : (byte >> 4);
    return (int32_t)nibble - (int32_t)kai_rhs_zero_point;
}

size_t kai_get_n_step_rhs_pack_kxn_qsi4c32ps4s0nrx4_qsu4c32s1s0(const size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_kxn_qsi4c32ps4s0nrx4_qsu4c32s1s0(
    const size_t n_idx,  //
    const size_t rhs_stride) {
    KAI_UNUSED(rhs_stride);
    KAI_ASSERT((n_idx % 2) == 0);

    return (n_idx / 2) * sizeof(int8_t);
}

size_t kai_get_rhs_packed_stride_rhs_pack_kxn_qsi4c32ps4s0nrx4_qsu4c32s1s0(
    // clang-format off
    const size_t k,
    const size_t nr,
    const size_t kr,
    const size_t sr,
    const size_t bl,
    const enum kai_datatype scale_dt) {
    // clang-format on
    KAI_ASSERT((k % bl) == 0);
    KAI_ASSERT((bl % kr) == 0);
    KAI_ASSERT((nr % kai_nr_multiple_of) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    KAI_ASSERT(scale_dt == kai_dt_bf16);

    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    const size_t num_bytes_multiplier_rhs = kai_get_datatype_size_in_bytes(scale_dt);
    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block(bl, num_bytes_multiplier_rhs);

    return nr * ((num_bytes_per_block * num_blocks_per_row) + kai_num_bytes_sum_rhs + kai_num_bytes_bias);
}

// clang-format off
size_t kai_get_rhs_packed_offset_rhs_pack_kxn_qsi4c32ps4s0nrx4_qsu4c32s1s0(
    const size_t n_idx,
    const size_t k,
    const size_t nr,
    const size_t kr,
    const size_t sr,
    const size_t bl,
    const enum kai_datatype scale_dt) {
    // clang-format on
    KAI_ASSERT((n_idx % nr) == 0);
    KAI_ASSERT((k % bl) == 0);
    KAI_ASSERT((bl % kr) == 0);
    KAI_ASSERT((nr % kai_nr_multiple_of) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    KAI_ASSERT(scale_dt == kai_dt_bf16);

    return (n_idx / nr) *
        kai_get_rhs_packed_stride_rhs_pack_kxn_qsi4c32ps4s0nrx4_qsu4c32s1s0(k, nr, kr, sr, bl, scale_dt);
}

// clang-format off
size_t kai_get_rhs_packed_size_rhs_pack_kxn_qsi4c32ps4s0nrx4_qsu4c32s1s0(
    const size_t n,   //
    const size_t k,   //
    const size_t nr,  //
    const size_t kr,  //
    const size_t sr,  //
    const size_t bl,  //
    const enum kai_datatype scale_dt) {
    // clang-format on
    KAI_ASSERT((k % bl) == 0);
    KAI_ASSERT((bl % kr) == 0);
    KAI_ASSERT((nr % kai_nr_multiple_of) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    KAI_ASSERT(scale_dt == kai_dt_bf16);

    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows *
        kai_get_rhs_packed_stride_rhs_pack_kxn_qsi4c32ps4s0nrx4_qsu4c32s1s0(k, nr, kr, sr, bl, scale_dt);
}

void kai_run_rhs_pack_kxn_qsi4c32ps4s0nrx4_qsu4c32s1s0(
    // clang-format off
    const size_t num_groups,
    const size_t n,
    const size_t k,
    const size_t nr,
    const size_t kr,
    const size_t sr,
    const size_t bl,
    const uint8_t* rhs,
    const size_t rhs_stride,
    const float* bias,
    const void* scale,
    const size_t scale_stride,
    void* rhs_packed,
    const size_t extra_bytes,
    const struct kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params* params) {
    // clang-format on
    KAI_UNUSED(num_groups);
    KAI_UNUSED(extra_bytes);
    KAI_ASSERT(rhs != NULL);
    KAI_ASSERT(scale != NULL);
    KAI_ASSERT(rhs_packed != NULL);
    KAI_ASSERT(params != NULL);
    KAI_ASSERT(params->rhs_zero_point == 8);
    KAI_ASSERT(params->lhs_zero_point == 1);

    KAI_ASSERT((k % bl) == 0);
    KAI_ASSERT((bl % kr) == 0);
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSERT((nr % kai_nr_multiple_of) == 0);
    KAI_ASSERT((bl % kai_bl_multiple_of) == 0);
    KAI_ASSERT(params->scale_dt == kai_dt_bf16);

    // Note: The input matrix (rhs) is expected with:
    // "k" rows and "n" columns (kxn)
    const size_t block_length = kr / sr;
    KAI_ASSERT(block_length == 4);
    const enum kai_datatype scale_dt = params->scale_dt;
    const size_t num_bytes_multiplier_rhs = kai_get_datatype_size_in_bytes(scale_dt);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block(bl, num_bytes_multiplier_rhs);
    const size_t rhs_packed_offset_end_of_all_blocks =
        kai_get_rhs_packed_offset_end_of_all_blocks(k, nr, kr, bl, num_bytes_multiplier_rhs);
    const size_t num_qblocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block_k = bl / 2;
    const size_t dst_num_rows = kai_roundup(n, nr);
    // Each s4s0 packet is 4 bytes (one uint32_t), covering 8 K-values -- matches
    // kai_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon's packet layout exactly, so the two
    // packers produce byte-identical packed RHS for the same (n, k, nr, kr, sr, bl).
    const size_t packet_bytes = block_length;
    const size_t num_packets_per_block = num_bytes_per_block_k / packet_bytes;

    uint8_t* dst_row_base = (uint8_t*)rhs_packed;

    for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; dst_row_idx += nr) {
        uint8_t* dst_row = dst_row_base;
        float* sums = (float*)(dst_row + rhs_packed_offset_end_of_all_blocks);

        // Initialize the RHS reduction sums to zero
        memset(sums, 0, nr * kai_num_bytes_sum_rhs);

        for (size_t dst_qblock_idx = 0; dst_qblock_idx < num_qblocks_per_row; ++dst_qblock_idx) {
            uint8_t* rhs_packed_scale = dst_row + num_bytes_per_block_k * nr;
            const uint8_t* scale_ptr = (const uint8_t*)scale + dst_qblock_idx * num_bytes_multiplier_rhs;

            for (size_t i = 0; i < nr; ++i) {
                const size_t src_col_idx = KAI_MIN(dst_row_idx + i, n - 1);
                const void* src_scales_ptr = scale_ptr + src_col_idx * scale_stride;
                void* dst_scales_ptr = rhs_packed_scale + i * num_bytes_multiplier_rhs;

                uint16_t src_scale_bits;
                memcpy(&src_scale_bits, src_scales_ptr, sizeof(src_scale_bits));
                const float src_scale_f32 = kai_cast_f32_bf16(src_scale_bits);

                // Store the pre-divided (by 1/16) scale for the kernel's lsl/and decode to
                // consume directly; the ORIGINAL, undivided value (src_scale_f32) is used below
                // for the rsum computation.
                const uint16_t dst_scale_bits = kai_cast_bf16_f32(src_scale_f32 * kai_pre_scaled_rhs_scale_factor);
                memcpy(dst_scales_ptr, &dst_scale_bits, sizeof(dst_scale_bits));

                const size_t k0_base = dst_qblock_idx * bl;
                float partial_sum = 0.0F;
                for (size_t k_off = 0; k_off < bl; ++k_off) {
                    const size_t k_idx = k0_base + k_off;
                    if (k_idx < k) {
                        partial_sum += (float)kai_read_signed_nibble_kxn(rhs, rhs_stride, k_idx, src_col_idx);
                    }
                }
                sums[i] += partial_sum * src_scale_f32;
            }

            // Pack the s4s0 K0/K4 nibble layout: packet p covers K[8p..8p+7] (relative to this
            // block's k0_base); byte b (0..3) holds K[8p+b] in the low nibble and K[8p+b+4] in
            // the high nibble, signed two's complement via XOR 0x88.
            const size_t k0_base = dst_qblock_idx * bl;
            for (size_t packet_idx = 0; packet_idx < num_packets_per_block; ++packet_idx) {
                for (size_t b = 0; b < packet_bytes; ++b) {
                    const size_t k_lo = k0_base + packet_idx * 8 + b;
                    const size_t k_hi = k_lo + 4;

                    for (size_t col_idx = 0; col_idx < nr; ++col_idx) {
                        const size_t n_idx = KAI_MIN(dst_row_idx + col_idx, n - 1);

                        const uint8_t lo_biased = (k_lo < k)
                            ? (uint8_t)(kai_read_signed_nibble_kxn(rhs, rhs_stride, k_lo, n_idx) + kai_rhs_zero_point)
                            : (uint8_t)kai_rhs_zero_point;
                        const uint8_t hi_biased = (k_hi < k)
                            ? (uint8_t)(kai_read_signed_nibble_kxn(rhs, rhs_stride, k_hi, n_idx) + kai_rhs_zero_point)
                            : (uint8_t)kai_rhs_zero_point;

                        const uint8_t packed_byte = (uint8_t)((lo_biased & 0x0F) | (hi_biased << 4)) ^ 0x88;

                        dst_row[packet_idx * nr * packet_bytes + col_idx * packet_bytes + b] = packed_byte;
                    }
                }
            }

            dst_row += num_bytes_per_block * nr;
        }

        // Move the pointer after the row sum
        dst_row += kai_num_bytes_sum_rhs * nr;

        // Set the bias
        if (bias == NULL) {
            memset(dst_row, 0, nr * kai_num_bytes_bias);
        } else {
            for (size_t i = 0; i < nr; ++i) {
                // Clamp the column index to avoid out-of-bound reads
                const size_t src_col_idx = KAI_MIN(dst_row_idx + i, n - 1);
                ((float*)dst_row)[i] = bias[src_col_idx];
            }
        }

        dst_row += kai_num_bytes_bias * nr;
        dst_row_base = dst_row;
    }
}
#endif  // Architectural features check.
