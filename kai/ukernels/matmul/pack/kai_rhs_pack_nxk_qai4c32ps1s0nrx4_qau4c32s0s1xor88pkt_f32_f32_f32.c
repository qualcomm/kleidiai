//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//

#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.
#include "kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1xor88pkt_f32_f32_f32.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_offset_rhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);
static const size_t kai_bl_multiple_of = 32;
static const size_t kai_nr_multiple_of = 4;
// K values per packet. Each packet is stored as one 4-byte group per N row; byte b holds
// K[8p + b] in its low nibble and K[8p + b + 4] in its high nibble. Chosen to match the
// 4-way byte reduction that both SDOT (indexed) and SMOPA perform.
static const size_t kai_packet_k = 8;
// XOR 0x88 converts the zero-point-8 nibble values to signed two's-complement.
static const uint8_t kai_sign_xor_value = 0x88;
// The micro-kernels' decode scales every value by 16 (top-justified byte read as signed);
// pre-dividing the block scale by 16 here compensates.
static const float kai_pre_scaled_rhs_scale_factor = 1.0f / 16.0f;

inline static size_t kai_get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((bl % kai_bl_multiple_of) == 0);
    return kai_roundup(k, bl) / bl;
}

inline static size_t kai_get_num_bytes_per_block(size_t bl) {
    return (bl / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_offset_rhs;
}

inline static size_t kai_get_rhs_packed_stride(size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((bl % kr) == 0);
    KAI_ASSUME((bl % kai_bl_multiple_of) == 0);
    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block(bl);
    return nr * (num_bytes_per_block * num_blocks_per_row + kai_num_bytes_bias);
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1xor88pkt_f32_f32_f32(
    size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1xor88pkt_f32_f32_f32(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((n_idx % nr) == 0);
    KAI_UNUSED(kr);
    return (n_idx / nr) * kai_get_rhs_packed_stride(k, nr, kr, bl);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1xor88pkt_f32_f32_f32(
    size_t n, size_t k, size_t nr, size_t kr, size_t bl) {
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_UNUSED(kr);
    const size_t num_rows = kai_roundup(n, nr) / nr;
    return num_rows * kai_get_rhs_packed_stride(k, nr, kr, bl);
}

// Packs one 8-K packet for a single N row: reads the 4 consecutive source bytes holding
// K[8p..8p+7] and writes the 4 packed bytes, where output byte b holds K[8p+b] in its low
// nibble and K[8p+b+4] in its high nibble, XOR-0x88'd.
//
// This is the closed form of the per-nibble definition. With the s0s1 source order (source
// column c holds k=2c in its HIGH nibble and k=2c+1 in its low nibble -- reversed relative to
// s1s0), packet p maps to source columns 4p..4p+3, and the four output bytes draw only on
// (s0,s2) and (s1,s3):
//   out0 = s0.hi | s2.hi<<4      out1 = s0.lo | s2.lo<<4
//   out2 = s1.hi | s3.hi<<4      out3 = s1.lo | s3.lo<<4
// i.e. the s1s0 form with each pair of outputs swapped.
inline static void kai_pack_packet(const uint8_t* src, uint8_t* dst) {
    const uint8_t s0 = src[0];
    const uint8_t s1 = src[1];
    const uint8_t s2 = src[2];
    const uint8_t s3 = src[3];

    dst[0] = (uint8_t)(((uint8_t)(s0 >> 4) | (s2 & 0xF0)) ^ kai_sign_xor_value);
    dst[1] = (uint8_t)(((s0 & 0x0F) | (uint8_t)((s2 & 0x0F) << 4)) ^ kai_sign_xor_value);
    dst[2] = (uint8_t)(((uint8_t)(s1 >> 4) | (s3 & 0xF0)) ^ kai_sign_xor_value);
    dst[3] = (uint8_t)(((s1 & 0x0F) | (uint8_t)((s3 & 0x0F) << 4)) ^ kai_sign_xor_value);
}

void kai_run_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1xor88pkt_f32_f32_f32(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, size_t bl, const uint8_t* rhs,
    const void* zero, const void* bias, const void* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qai4c32p_params* params) {
    KAI_ASSUME(num_groups == 1);
    KAI_ASSUME((k % 2) == 0);
    KAI_ASSUME((k % kr) == 0);
    KAI_ASSUME((k % bl) == 0);
    KAI_ASSUME((bl % kai_bl_multiple_of) == 0);
    KAI_ASSUME((nr % kai_nr_multiple_of) == 0);
    KAI_ASSUME(extra_bytes == 0);

    KAI_ASSUME(sr == 2);
    KAI_ASSUME(kr / sr == 4);
    KAI_ASSUME(rhs != NULL);
    KAI_ASSUME(zero != NULL);
    KAI_ASSUME(scale != NULL);
    KAI_ASSUME(rhs_packed != NULL);
    KAI_ASSUME(params != NULL);
    KAI_ASSUME(params->rhs_zero_point == 8);
    KAI_ASSUME(params->lhs_zero_point == 1);

    // Note: The input matrix (rhs) is expected with:
    // "k" columns and "n" rows (NxK)

    const size_t num_blocks_per_row = k / bl;
    const size_t rhs_stride = k / 2;
    const size_t rhs_packed_stride = kai_get_rhs_packed_stride(k, nr, kr, bl);

    const size_t dst_packed_block_size = kai_get_num_bytes_per_block(bl) * nr;
    const size_t dst_block_data_size = bl / 2;
    const size_t dst_num_rows = kai_roundup(n, nr) / nr;
    const size_t dst_bias_offset = num_blocks_per_row * dst_packed_block_size;

    const size_t packets_per_block = bl / kai_packet_k;

    for (size_t dst_row_idx = 0; dst_row_idx < dst_num_rows; ++dst_row_idx) {
        uint8_t* dst_row = (uint8_t*)rhs_packed + dst_row_idx * rhs_packed_stride;
        float* dst_row_bias = (float*)(dst_row + dst_bias_offset);
        size_t row_idx = dst_row_idx * nr;
        size_t rows_left = n - row_idx;

        for (size_t block_idx = 0; block_idx < num_blocks_per_row; block_idx++) {
            uint8_t* block_dst_row = dst_row + block_idx * dst_packed_block_size;
            float* block_dst_zp = (float*)(block_dst_row + nr * dst_block_data_size);
            float* block_dst_scale = block_dst_zp + nr;
            const size_t block_k_base = block_idx * bl;

            // Packet-major, N-minor: 32-bit slot (p * nr + n_local) holds packet p of row n_local.
            // Packet p covers K[8p..8p+7], i.e. source columns 4p..4p+3 -- 4 consecutive bytes --
            // so each packet is one short contiguous read and one 4-byte write.
            for (size_t p = 0; p < packets_per_block; ++p) {
                const size_t src_byte_base = (block_k_base + p * kai_packet_k) / 2;

                for (size_t n_local = 0; n_local < nr; ++n_local) {
                    const size_t n0_idx = KAI_MIN(row_idx + n_local, n - 1);

                    kai_pack_packet(
                        rhs + n0_idx * rhs_stride + src_byte_base, block_dst_row + (p * nr + n_local) * 4);
                }
            }

            // Adjust the zero points and scales
            for (size_t i = 0; i < nr; ++i) {
                const size_t src_row_idx = KAI_MIN(row_idx + i, n - 1);
                const size_t src_idx = src_row_idx * num_blocks_per_row + block_idx;

                block_dst_scale[i] = ((const float*)scale)[src_idx] * kai_pre_scaled_rhs_scale_factor;
                block_dst_zp[i] = ((const float*)zero)[src_idx];
            }
        }
        // Set the bias
        if (bias == NULL) {
            memset(dst_row_bias, 0, nr * kai_num_bytes_bias);
        } else {
            if (rows_left >= nr) {
                memcpy(dst_row_bias, &((const float*)bias)[row_idx], nr * kai_num_bytes_bias);
            } else {
                // Fill remaining values
                memcpy(dst_row_bias, &((const float*)bias)[row_idx], rows_left * kai_num_bytes_bias);
                // Set leftover to 0
                memset(&dst_row_bias[rows_left], 0, (nr - rows_left) * kai_num_bytes_bias);
            }
        }
    }
}
#endif
