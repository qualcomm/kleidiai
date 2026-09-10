//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

extern void kai_run_rhs_pack_nxk_qsi4cxp8vsx4s4s0sf32bi32_qsx4cx_f32_i32_sme(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args,
    int32_t rhs_zero_point);

enum {
    BIAS_ELEM_BYTES = sizeof(int32_t),
    SCALE_ELEM_BYTES = sizeof(float),
    RHS_ELEM_RECIP_BYTES = 2,

    NR_VSCALE = 8,
    NR_TILE = 4,
    K_MULTIPLE = 32,
    K_SPAN = 16,  // K values rearranged per NEON pass (two 8-K packets)

    MAX_NR = NR_VSCALE * KAI_VSCALE_MAX,
};

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

/// Pair 16 unpacked nibbles K[0..15] into 8 bytes with K+0/K+4 pairing:
///   out[b]     = K[b]     | K[b + 4]  << 4   (packet 0, K[0..7])
///   out[4 + b] = K[8 + b] | K[12 + b] << 4   (packet 1, K[8..15])
///
/// Same transform and table indices as pack_pairs_from_nibbles() in
/// kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon.c, which produces the identical "s4s0"
/// packet format for the blockwise f16-scale container. Duplicated rather than shared because
/// pack helpers in this directory are file-local by convention.
static inline uint8x8_t kai_pair_nibbles_k4(const uint8x16_t nibbles) {
    static const uint8_t idx_lo_tbl[8] = {0, 1, 2, 3, 8, 9, 10, 11};
    static const uint8_t idx_hi_tbl[8] = {4, 5, 6, 7, 12, 13, 14, 15};
    const uint8x8_t lo = vqtbl1_u8(nibbles, vld1_u8(idx_lo_tbl));
    const uint8x8_t hi = vqtbl1_u8(nibbles, vld1_u8(idx_hi_tbl));
    return vorr_u8(lo, vshl_n_u8(hi, 4));
}

/// Signed row-sum contribution of 16 unpacked nibbles, used for the LHS zero-point correction.
/// Values are first mapped to offset-binary (value + 8) -- qsi4cx nibbles arrive two's-complement
/// (xor 8), qsu4cx nibbles are already offset-binary -- summed widening, then de-biased.
/// Mirrors the vaddlvq_u8 idiom in kai_rhs_pack_nxk_qsi4cxp8vsx4sf32bi32_qsx4cx_f32_i32_sme.c.
static inline int32_t kai_nibble_sum_k16(const uint8x16_t nibbles, const uint8_t offset_binary_flip) {
    const uint8x16_t offset_binary = veorq_u8(nibbles, vdupq_n_u8(offset_binary_flip));
    return (int32_t)vaddlvq_u8(offset_binary) - 8 * K_SPAN;
}

/// Nibble (n, k) of an NxK i4 matrix: row-major over n, two nibbles per byte along k.
static inline uint8_t kai_rhs_nibble_nxk(const uint8_t* rhs, size_t n_idx, size_t stride, size_t k_idx) {
    const uint8_t byte = rhs[n_idx * stride + k_idx / 2];
    return (k_idx % 2 == 0) ? (uint8_t)(byte & 0x0FU) : (uint8_t)(byte >> 4);
}

/// "s4s0" payload layout, for QMX kernels that decode nibbles arithmetically.
///
/// Container (bias | payload | scale), every stride and every offset is byte-identical to
/// @ref kai_run_rhs_pack_nxk_qsi4cxp8vsx4sf32bi32_qsx4cx_f32_i32_sme, and the bias/scale
/// epilogue below is that packer's, unchanged. Only the arrangement of nibbles WITHIN the payload
/// differs, so the two are NOT interchangeable and a mismatch is not rejected by any assert -- it
/// silently produces wrong results. See the warning on the consuming kernel in kai_matmul.h.
///
/// Each 32-K block occupies 8 vectors of nr*2 bytes. Vector r = kg8 * 2 + n_half holds nr/2
/// columns; column j of that half sits at bytes 4j..4j+3, and byte 4j+i packs
/// K[32b + 8*kg8 + i] in its low nibble and K[32b + 8*kg8 + 4 + i] in its high nibble.
/// Stored nibbles are two's-complement signed i4, so one `lsl #4; asr #4` yields the low plane
/// and one `asr #4` the high plane -- each already a valid SMOPA operand, with no zip needed.
///
/// Padding (columns beyond n, K beyond k) encodes to 0x00 for both supported zero points, because
/// the padded nibble pair (zp | zp << 4) is exactly what the qsu4cx sign flip maps to zero. The
/// payload is therefore zeroed once per panel and only in-range data is written.
void kai_run_rhs_pack_nxk_qsi4cxp8vsx4s4s0sf32bi32_qsx4cx_f32_i32_sme(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args,
    const int32_t rhs_zero_point) {
    KAI_ASSUME(args->flags == 0);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs_packed.ptr != NULL);
    KAI_ASSUME(args->operand.bias_n.ptr != NULL);
    KAI_ASSUME(args->operand.k_sum_scale_global.ptr != NULL);
    KAI_ASSUME(args->operand.scale_n.ptr != NULL);
    KAI_ASSUME(args->operand.scale_global.ptr != NULL);
    KAI_ASSUME(rhs_zero_point == 0 || rhs_zero_point == 8);
    KAI_UNUSED(config);

    const size_t n = args->shape.n;
    const size_t k = args->shape.k;
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);

    const size_t rhs_stride_row = args->operand.rhs.stride.n;
    const int32_t k_sum_scale = *(const int32_t*)args->operand.k_sum_scale_global.ptr;
    const float scale_multiplier = *(const float*)args->operand.scale_global.ptr;

    if (n == 0 || k == 0) {
        return;
    }

    const size_t rounded_k = kai_roundup(k, K_MULTIPLE);
    const size_t packed_stride = args->operand.rhs_packed.stride.n;
    const size_t payload_bytes = rounded_k * nr / RHS_ELEM_RECIP_BYTES;
    const size_t half = nr / 2;
    const size_t vec_bytes = nr * 2;  // one payload vector == one SVE vector == nr*2 bytes
    const uint8_t* rhs_ptr = (const uint8_t*)args->operand.rhs.ptr;
    const int32_t* bias_ptr = (const int32_t*)args->operand.bias_n.ptr;
    const float* scale_ptr = (const float*)args->operand.scale_n.ptr;
    uint8_t* packed_ptr = (uint8_t*)args->operand.rhs_packed.ptr;

    // Stored nibbles are two's-complement: qsu4cx input needs the flip, qsi4cx is already signed.
    const uint8_t store_flip = (rhs_zero_point == 0) ? 0x00U : 0x88U;
    // Offset-binary mapping for the row sums is the complementary case.
    const uint8_t sum_flip = (rhs_zero_point == 0) ? 0x08U : 0x00U;

    for (size_t n_base = 0; n_base < n; n_base += nr) {
        const size_t block_width = KAI_MIN(n - n_base, nr);
        uint8_t* payload = packed_ptr + nr * BIAS_ELEM_BYTES;
        uint8_t* scale_out = payload + payload_bytes;
        int32_t sums[MAX_NR];

        // Covers every padded column and every padded K in one pass; see note above.
        memset(payload, 0, payload_bytes);

        for (size_t group = 0; group < block_width; ++group) {
            const size_t n_idx = n_base + group;
            const size_t n_half = group / half;
            const size_t j = group % half;
            int32_t sum = 0;

            for (size_t k_base = 0; k_base < k; k_base += K_SPAN) {
                uint8x16_t nibbles;

                if (k_base + K_SPAN <= k) {
                    // Fast path: byte q of this span holds K[k_base + 2q] low and K[k_base + 2q + 1]
                    // high, so masking and zipping the two nibble planes rebuilds K[0..15] in order.
                    const uint8x8_t raw = vld1_u8(rhs_ptr + n_idx * rhs_stride_row + k_base / 2);
                    const uint8x8x2_t planes = vzip_u8(vand_u8(raw, vdup_n_u8(0x0FU)), vshr_n_u8(raw, 4));
                    nibbles = vcombine_u8(planes.val[0], planes.val[1]);
                } else {
                    uint8_t tail[K_SPAN];
                    for (size_t t = 0; t < K_SPAN; ++t) {
                        const size_t k_idx = k_base + t;
                        tail[t] = (k_idx < k) ? kai_rhs_nibble_nxk(rhs_ptr, n_idx, rhs_stride_row, k_idx)
                                              : (uint8_t)rhs_zero_point;
                    }
                    nibbles = vld1q_u8(tail);
                }

                sum += kai_nibble_sum_k16(nibbles, sum_flip);

                const uint8x8_t paired = veor_u8(kai_pair_nibbles_k4(nibbles), vdup_n_u8(store_flip));
                uint8_t packets[8];
                vst1_u8(packets, paired);

                // The two 8-K packets of this span belong to consecutive kg8 groups.
                uint8_t* block_out = payload + (k_base / K_MULTIPLE) * 8 * vec_bytes;
                const size_t kg8_base = (k_base % K_MULTIPLE) / 8;
                for (size_t p = 0; p < 2; ++p) {
                    uint8_t* vec_out = block_out + ((kg8_base + p) * 2 + n_half) * vec_bytes;
                    memcpy(vec_out + j * 4, packets + p * 4, 4);
                }
            }

            sums[group] = sum;
        }

        // Bias/scale epilogue, unchanged from kai_run_rhs_pack_nxk_qsi4cxp8vsx4sf32bi32_qsx4cx.
        size_t group = 0;
        for (; group + NR_TILE <= block_width; group += NR_TILE) {
            const int32x4_t input_bias = vld1q_s32(bias_ptr + n_base + group);
            const int32x4_t rhs_sums = vld1q_s32(sums + group);
            const int32x4_t packed_bias = vmlaq_n_s32(input_bias, rhs_sums, k_sum_scale);
            vst1q_s32((int32_t*)(packed_ptr + group * BIAS_ELEM_BYTES), packed_bias);

            const float32x4_t input_scale = vld1q_f32(scale_ptr + n_base + group);
            vst1q_f32((float*)(scale_out + group * SCALE_ELEM_BYTES), vmulq_n_f32(input_scale, scale_multiplier));
        }

        for (; group < block_width; ++group) {
            ((int32_t*)packed_ptr)[group] = bias_ptr[n_base + group] + k_sum_scale * sums[group];
            ((float*)scale_out)[group] = scale_ptr[n_base + group] * scale_multiplier;
        }

        for (; group < nr; ++group) {
            ((int32_t*)packed_ptr)[group] = 0;
            ((float*)scale_out)[group] = 0.0F;
        }

        packed_ptr += packed_stride;
    }
}

#endif  // Architectural features check.
