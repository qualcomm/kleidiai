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

#include "kai_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon.h"

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_sum_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);
static const size_t kai_nr_multiple_of = 4;
static const size_t kai_bl_multiple_of = 32;

// The lsl/and decode used by the consuming matmul kernel leaves both nibbles in the upper 4
// bits of the byte (unshifted), which is numerically 16x too large. This is compensated by
// pre-dividing the stored per-block scale by 1/16 at pack time (mirroring
// kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon's kai_pre_scaled_rhs_scale_factor).
// The reduction-sum (rsum) computation below must use the ORIGINAL, undivided scale -- rsum
// reflects the true dequantized weight sum for the LHS zero-point correction and has nothing
// to do with how the kernel later decodes packed nibbles.
static const float kai_pre_scaled_rhs_scale_factor = 1.0F / 16.0F;

// Table-lookup indices mapping the s4s0 K0/K4 packet layout directly from a sequential (s1s0)
// source. For a 32-K-value chunk, lo[i] = K(2i), hi[i] = K(2i+1) (i = 0..15). Flattened into a
// combined 32-lane space (0..15 selects from lo, 16..31 selects from hi), idx_low[o]/idx_high[o]
// give the (low nibble, high nibble) source lanes for packed output byte o (o = 4p+b, packet p,
// byte b). See the header doc comment for the resulting byte layout.
static const uint8_t kai_s4s0_idx_low_tbl[16] = {0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29};
static const uint8_t kai_s4s0_idx_high_tbl[16] = {2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31};

// Packs one column's 16 signed nibble bytes (32 K-values, already zero-point-subtracted) into
// the s4s0 K0/K4 packet layout, directly from the sequential lo/hi nibble vectors.
static inline uint8x16_t kai_pack_s4s0_from_s1s0_nibbles(
    int8x16_t lo, int8x16_t hi, uint8x16_t idx_low, uint8x16_t idx_high, uint8x16_t low_mask) {
    uint8x16x2_t table;
    table.val[0] = vreinterpretq_u8_s8(lo);
    table.val[1] = vreinterpretq_u8_s8(hi);
    const uint8x16_t low_sel = vqtbl2q_u8(table, idx_low);
    const uint8x16_t high_sel = vqtbl2q_u8(table, idx_high);
    return vorrq_u8(vandq_u8(low_sel, low_mask), vshlq_n_u8(high_sel, 4));
}

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

size_t kai_get_n_step_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(const size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(const size_t n_idx, const size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(
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
size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(
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
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(k, nr, kr, sr, bl, scale_dt);
}

// clang-format off
size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(
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
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(k, nr, kr, sr, bl, scale_dt);
}

void kai_run_rhs_pack_nxk_qsi4c32ps4s0nrx4_qsu4c32s1s0_neon(
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
    const struct kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params* params) {
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
    // "k" columns and "n" rows (NxK)
    const size_t block_length = kr / sr;
    KAI_ASSERT(block_length == 4);
    const enum kai_datatype scale_dt = params->scale_dt;
    const size_t num_bytes_multiplier_rhs = kai_get_datatype_size_in_bytes(scale_dt);
    const size_t rhs_packed_offset_end_of_all_blocks =
        kai_get_rhs_packed_offset_end_of_all_blocks(k, nr, kr, bl, num_bytes_multiplier_rhs);
    const size_t num_qblocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block_k = bl / 2;
    const size_t dst_num_rows = kai_roundup(n, nr);
    // Each s4s0 packet is 4 bytes (one uint32_t), covering 8 K-values -- this is the
    // per-column payload unit that gets interleaved across the nr columns below.
    const size_t packet_bytes = block_length;

    const int8x16_t rhs_zero_point = vdupq_n_s8(8);
    const uint8x16_t low_mask = vdupq_n_u8(0x0F);
    const uint8x16_t idx_low = vld1q_u8(kai_s4s0_idx_low_tbl);
    const uint8x16_t idx_high = vld1q_u8(kai_s4s0_idx_high_tbl);
    const size_t num_bytes_processed = 16;

    uint8_t* dst_row = (uint8_t*)rhs_packed;

    for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; dst_row_idx += nr) {
        float* sums = (float*)(dst_row + rhs_packed_offset_end_of_all_blocks);

        // Initialize the RHS reduction sums to zero
        memset(sums, 0, nr * kai_num_bytes_sum_rhs);

        // Iterate over the quantized blocks
        for (size_t dst_qblock_idx = 0; dst_qblock_idx < num_qblocks_per_row; ++dst_qblock_idx) {
            // Store the scales after packing all K values in the block
            uint8_t* rhs_packed_scale = dst_row + num_bytes_per_block_k * nr;
            const uint8_t* scale_ptr = (const uint8_t*)scale + dst_qblock_idx * num_bytes_multiplier_rhs;

            for (size_t i = 0; i < nr; ++i) {
                const size_t src_row_idx = KAI_MIN(dst_row_idx + i, n - 1);
                const void* src_scales_ptr = scale_ptr + src_row_idx * scale_stride;
                void* dst_scales_ptr = rhs_packed_scale + i * num_bytes_multiplier_rhs;

                // Store the pre-divided (by 1/16) scale for the kernel's lsl/and decode to
                // consume directly; the original, undivided value is read straight from
                // src_scales_ptr below for the rsum computation.
                uint16_t src_scale_bits;
                memcpy(&src_scale_bits, src_scales_ptr, sizeof(src_scale_bits));
                const uint16_t dst_scale_bits =
                    kai_cast_bf16_f32(kai_cast_f32_bf16(src_scale_bits) * kai_pre_scaled_rhs_scale_factor);
                memcpy(dst_scales_ptr, &dst_scale_bits, sizeof(dst_scale_bits));
            }

            size_t k0_idx_i = dst_qblock_idx * bl;

            for (size_t dst_byte_idx = 0; dst_byte_idx < num_bytes_per_block_k; dst_byte_idx += num_bytes_processed) {
                for (size_t nr_idx = 0; nr_idx < nr; nr_idx += 4) {
                    // Clamp the indices to avoid out-of-bound reads
                    const size_t n0_idx = KAI_MIN(dst_row_idx + nr_idx, n - 1);
                    const size_t n1_idx = KAI_MIN(n0_idx + 1, n - 1);
                    const size_t n2_idx = KAI_MIN(n0_idx + 2, n - 1);
                    const size_t n3_idx = KAI_MIN(n0_idx + 3, n - 1);

                    // Load scales for the rsum computation from the ORIGINAL source scale
                    // buffer (not the packed buffer, which now holds the pre-divided value).
                    uint16_t d0_bits;
                    uint16_t d1_bits;
                    uint16_t d2_bits;
                    uint16_t d3_bits;
                    memcpy(&d0_bits, scale_ptr + n0_idx * scale_stride, sizeof(d0_bits));
                    memcpy(&d1_bits, scale_ptr + n1_idx * scale_stride, sizeof(d1_bits));
                    memcpy(&d2_bits, scale_ptr + n2_idx * scale_stride, sizeof(d2_bits));
                    memcpy(&d3_bits, scale_ptr + n3_idx * scale_stride, sizeof(d3_bits));
                    const float d0 = kai_cast_f32_bf16(d0_bits);
                    const float d1 = kai_cast_f32_bf16(d1_bits);
                    const float d2 = kai_cast_f32_bf16(d2_bits);
                    const float d3 = kai_cast_f32_bf16(d3_bits);

                    // Initialize partial sum
                    int32_t partial_sum0 = 0;
                    int32_t partial_sum1 = 0;
                    int32_t partial_sum2 = 0;
                    int32_t partial_sum3 = 0;

                    const uint8_t* src_block_base = rhs + ((k0_idx_i / 2) + dst_byte_idx);
                    const uint8x16_t vsrc0_0 = vld1q_u8(src_block_base + n0_idx * rhs_stride);
                    const uint8x16_t vsrc1_0 = vld1q_u8(src_block_base + n1_idx * rhs_stride);
                    const uint8x16_t vsrc2_0 = vld1q_u8(src_block_base + n2_idx * rhs_stride);
                    const uint8x16_t vsrc3_0 = vld1q_u8(src_block_base + n3_idx * rhs_stride);

                    // Get the lower and higher nibble and apply zero-points
                    const int8x16_t vsrc0_0_lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vsrc0_0, low_mask)), rhs_zero_point);
                    const int8x16_t vsrc0_0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vsrc0_0, 4)), rhs_zero_point);
                    const int8x16_t vsrc1_0_lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vsrc1_0, low_mask)), rhs_zero_point);
                    const int8x16_t vsrc1_0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vsrc1_0, 4)), rhs_zero_point);
                    const int8x16_t vsrc2_0_lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vsrc2_0, low_mask)), rhs_zero_point);
                    const int8x16_t vsrc2_0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vsrc2_0, 4)), rhs_zero_point);
                    const int8x16_t vsrc3_0_lo =
                        vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vsrc3_0, low_mask)), rhs_zero_point);
                    const int8x16_t vsrc3_0_hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vsrc3_0, 4)), rhs_zero_point);

                    // Calculate and store row sums
                    partial_sum0 += vaddlvq_s16(vaddl_s8(
                        vadd_s8(vget_low_s8(vsrc0_0_lo), vget_high_s8(vsrc0_0_lo)),
                        vadd_s8(vget_low_s8(vsrc0_0_hi), vget_high_s8(vsrc0_0_hi))));
                    partial_sum1 += vaddlvq_s16(vaddl_s8(
                        vadd_s8(vget_low_s8(vsrc1_0_lo), vget_high_s8(vsrc1_0_lo)),
                        vadd_s8(vget_low_s8(vsrc1_0_hi), vget_high_s8(vsrc1_0_hi))));
                    partial_sum2 += vaddlvq_s16(vaddl_s8(
                        vadd_s8(vget_low_s8(vsrc2_0_lo), vget_high_s8(vsrc2_0_lo)),
                        vadd_s8(vget_low_s8(vsrc2_0_hi), vget_high_s8(vsrc2_0_hi))));
                    partial_sum3 += vaddlvq_s16(vaddl_s8(
                        vadd_s8(vget_low_s8(vsrc3_0_lo), vget_high_s8(vsrc3_0_lo)),
                        vadd_s8(vget_low_s8(vsrc3_0_hi), vget_high_s8(vsrc3_0_hi))));

                    // NOLINTBEGIN(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
                    sums[nr_idx + 0] += (float)partial_sum0 * d0;
                    sums[nr_idx + 1] += (float)partial_sum1 * d1;
                    sums[nr_idx + 2] += (float)partial_sum2 * d2;
                    sums[nr_idx + 3] += (float)partial_sum3 * d3;
                    // NOLINTEND(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)

                    // Pack the s4s0 K0/K4 nibble layout directly from the sequential (s1s0)
                    // lo/hi nibble vectors above (this is the one change vs.
                    // kai_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon, which instead
                    // re-packs lo/hi back into sequential nibble pairs here).
                    const uint8x16_t vdst_u8_0 =
                        kai_pack_s4s0_from_s1s0_nibbles(vsrc0_0_lo, vsrc0_0_hi, idx_low, idx_high, low_mask);
                    const uint8x16_t vdst_u8_1 =
                        kai_pack_s4s0_from_s1s0_nibbles(vsrc1_0_lo, vsrc1_0_hi, idx_low, idx_high, low_mask);
                    const uint8x16_t vdst_u8_2 =
                        kai_pack_s4s0_from_s1s0_nibbles(vsrc2_0_lo, vsrc2_0_hi, idx_low, idx_high, low_mask);
                    const uint8x16_t vdst_u8_3 =
                        kai_pack_s4s0_from_s1s0_nibbles(vsrc3_0_lo, vsrc3_0_hi, idx_low, idx_high, low_mask);

                    // Reorder to interleave nr columns at packet (4-byte) granularity: each
                    // vdst_u8_c holds 4 packets (uint32 each) for column c. This differs from
                    // kai_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon's byte->u16->u32 zip
                    // chain, which was built for its 2-byte-per-column payload; here the
                    // per-column payload is already a native uint32 packet, so only ONE level
                    // of zip is needed (no u16 intermediate), otherwise a 4-byte packet would be
                    // split in half and scattered across two different, wrongly-mixed slots.
                    const uint32x4_t vpacket_0 = vreinterpretq_u32_u8(vdst_u8_0);
                    const uint32x4_t vpacket_1 = vreinterpretq_u32_u8(vdst_u8_1);
                    const uint32x4_t vpacket_2 = vreinterpretq_u32_u8(vdst_u8_2);
                    const uint32x4_t vpacket_3 = vreinterpretq_u32_u8(vdst_u8_3);

                    const uint32x4_t vzip_01_lo = vzip1q_u32(vpacket_0, vpacket_1);
                    const uint32x4_t vzip_23_lo = vzip1q_u32(vpacket_2, vpacket_3);
                    const uint32x4_t vzip_01_hi = vzip2q_u32(vpacket_0, vpacket_1);
                    const uint32x4_t vzip_23_hi = vzip2q_u32(vpacket_2, vpacket_3);

                    // packet_row_p = [column0's packet p, column1's packet p, column2's packet
                    // p, column3's packet p], for p = 0..3.
                    const uint32x4_t packet_row_0 = vcombine_u32(vget_low_u32(vzip_01_lo), vget_low_u32(vzip_23_lo));
                    const uint32x4_t packet_row_1 = vcombine_u32(vget_high_u32(vzip_01_lo), vget_high_u32(vzip_23_lo));
                    const uint32x4_t packet_row_2 = vcombine_u32(vget_low_u32(vzip_01_hi), vget_low_u32(vzip_23_hi));
                    const uint32x4_t packet_row_3 = vcombine_u32(vget_high_u32(vzip_01_hi), vget_high_u32(vzip_23_hi));

                    // Store packed values: one packet-row (4 columns' packet p) per slot,
                    // slots spaced nr * packet_bytes apart.
                    vst1q_u32((uint32_t*)dst_row, packet_row_0);
                    vst1q_u32((uint32_t*)(dst_row + nr * packet_bytes), packet_row_1);
                    vst1q_u32((uint32_t*)(dst_row + 2 * nr * packet_bytes), packet_row_2);
                    vst1q_u32((uint32_t*)(dst_row + 3 * nr * packet_bytes), packet_row_3);

                    dst_row += (4 * packet_bytes);
                }
                // Skip to end of qblock (3 remaining packet-row slots, each nr * packet_bytes wide).
                dst_row += 3 * nr * packet_bytes;
            }

            // Move the pointer after scales
            dst_row += num_bytes_multiplier_rhs * nr;
        }

        // Move the pointer after the row sum
        dst_row += kai_num_bytes_sum_rhs * nr;

        // Set the bias
        if (bias == NULL) {
            memset(dst_row, 0, nr * kai_num_bytes_bias);
        } else {
            for (size_t i = 0; i < nr; ++i) {
                // Clamp the row index to avoid out-of-bound reads
                const size_t src_row_idx = KAI_MIN(dst_row_idx + i, n - 1);
                ((float*)dst_row)[i] = bias[src_row_idx];
            }
        }

        // Move the pointer after the row sum
        dst_row += kai_num_bytes_bias * nr;
    }
}
#endif  // Architectural features check.
