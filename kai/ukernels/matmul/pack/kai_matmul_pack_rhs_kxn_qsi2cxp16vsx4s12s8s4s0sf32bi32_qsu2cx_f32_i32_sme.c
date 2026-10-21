//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

enum {
    BIAS_ELEM_BYTES = sizeof(int32_t),
    SCALE_ELEM_BYTES = sizeof(float),
    RHS_ELEM_RECIP_BYTES = 4,

    NR_VSCALE = 16,
    NR_TILE = 4,
    K_MULTIPLE = 32,
    K_UNIT = 16,

    RHS_ZERO_POINT = 2,
    MAX_NR = NR_VSCALE * KAI_VSCALE_MAX,
};

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

static struct kai_matmul_pack_rhs_uker_dim_args get_step(const struct kai_matmul_pack_rhs_uker_config* config) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr % RHS_ELEM_RECIP_BYTES == 0);
    KAI_ASSUME(nr <= MAX_NR);
    const struct kai_matmul_pack_rhs_uker_dim_args step = {
        .n = nr,
        .k = 0,
    };
    return step;
}

static struct kai_matmul_pack_rhs_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args stride = {
        .n = 0,
        .k = kai_roundup(shape->n, RHS_ELEM_RECIP_BYTES) / RHS_ELEM_RECIP_BYTES,
    };
    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_UNUSED(stride);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr % RHS_ELEM_RECIP_BYTES == 0);
    KAI_ASSUME(nr <= MAX_NR);
    KAI_ASSUME(index->n % nr == 0);
    KAI_ASSUME(index->k == 0);
    return index->n / RHS_ELEM_RECIP_BYTES;
}

static struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args get_rhs_packed_stride(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride = {
        .n = nr * (BIAS_ELEM_BYTES + kai_roundup(shape->k, K_MULTIPLE) / RHS_ELEM_RECIP_BYTES + SCALE_ELEM_BYTES),
    };
    return stride;
}

static size_t get_rhs_packed_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);
    KAI_ASSUME(index->n % nr == 0);
    KAI_ASSUME(index->k == 0);
    return index->n / nr * stride->n;
}

static size_t get_rhs_packed_size(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);
    return kai_roundup(shape->n, nr) / nr * stride->n;
}

static size_t get_bias_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_bias_n_dim_args* index) {
    KAI_UNUSED(config);
    return index->n * BIAS_ELEM_BYTES;
}

static size_t get_scale_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_scale_n_dim_args* index) {
    KAI_UNUSED(config);
    return index->n * SCALE_ELEM_BYTES;
}

static uint8_t load_qsu2(const uint8_t* rhs, size_t rhs_stride, size_t n_idx, size_t k_idx, size_t n, size_t k) {
    if (n_idx >= n || k_idx >= k) {
        return RHS_ZERO_POINT;
    }

    const uint8_t value = rhs[k_idx * rhs_stride + n_idx / RHS_ELEM_RECIP_BYTES];
    return (value >> ((n_idx % RHS_ELEM_RECIP_BYTES) * 2)) & 0x03;
}

static uint8x16_t expand_qsu2(uint32_t input) {
    static const uint8_t table_indices[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    static const int8_t shifts[16] = {0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6};
    const uint8x16_t input_vector = vcombine_u8(vcreate_u8(input), vdup_n_u8(0));
    const uint8x16_t repeated = vqtbl1q_u8(input_vector, vld1q_u8(table_indices));
    return vandq_u8(vshlq_u8(repeated, vld1q_s8(shifts)), vdupq_n_u8(0x03));
}

static void accumulate_qsu2_sum(int32_t* sums, const uint8x16_t raw_sum) {
    const uint8x8_t zero_point_sum = vdup_n_u8(RHS_ZERO_POINT * K_UNIT);
    const int16x8_t low = vreinterpretq_s16_u16(vsubl_u8(vget_low_u8(raw_sum), zero_point_sum));
    const int16x8_t high = vreinterpretq_s16_u16(vsubl_u8(vget_high_u8(raw_sum), zero_point_sum));

    vst1q_s32(sums, vaddq_s32(vld1q_s32(sums), vmovl_s16(vget_low_s16(low))));
    vst1q_s32(sums + 4, vaddq_s32(vld1q_s32(sums + 4), vmovl_s16(vget_high_s16(low))));
    vst1q_s32(sums + 8, vaddq_s32(vld1q_s32(sums + 8), vmovl_s16(vget_low_s16(high))));
    vst1q_s32(sums + 12, vaddq_s32(vld1q_s32(sums + 12), vmovl_s16(vget_high_s16(high))));
}

/// Packs signed s12s8s4s0 packets from a KxN QSU2 matrix.
///
/// Within each 16-K unit, row n owns four consecutive bytes. The four QSI2 fields in byte r are
/// K+r, K+4+r, K+8+r, and K+12+r from least to most significant.
static void run(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME(args->flags == 0);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs_packed.ptr != NULL);
    KAI_ASSUME(args->operand.bias_n.ptr != NULL);
    KAI_ASSUME(args->operand.k_sum_scale_global.ptr != NULL);
    KAI_ASSUME(args->operand.scale_n.ptr != NULL);
    KAI_ASSUME(args->operand.scale_global.ptr != NULL);

    const size_t n = args->shape.n;
    const size_t k = args->shape.k;
    if (n == 0 || k == 0) {
        return;
    }

    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);
    const size_t rounded_k = kai_roundup(k, K_MULTIPLE);
    const size_t rhs_stride = args->operand.rhs.stride.k;
    const size_t packed_stride = args->operand.rhs_packed.stride.n;
    const int32_t k_sum_scale = *(const int32_t*)args->operand.k_sum_scale_global.ptr;
    const float scale_multiplier = *(const float*)args->operand.scale_global.ptr;

    const uint8_t* rhs_ptr = (const uint8_t*)args->operand.rhs.ptr;
    const int32_t* bias_ptr = (const int32_t*)args->operand.bias_n.ptr;
    const float* scale_ptr = (const float*)args->operand.scale_n.ptr;
    uint8_t* packed_ptr = (uint8_t*)args->operand.rhs_packed.ptr;

    for (size_t n_base = 0; n_base < n; n_base += nr) {
        const size_t block_width = KAI_MIN(n - n_base, nr);
        int32_t rhs_sums[MAX_NR] = {0};
        int32_t* packed_bias = (int32_t*)packed_ptr;
        uint8_t* packed_rhs = packed_ptr + nr * BIAS_ELEM_BYTES;
        float* packed_scale = (float*)(packed_rhs + nr * rounded_k / RHS_ELEM_RECIP_BYTES);

        memset(packed_rhs, 0, nr * rounded_k / RHS_ELEM_RECIP_BYTES);

        for (size_t k_base = 0; k_base < rounded_k; k_base += K_UNIT) {
            uint8_t* const packed_unit = packed_rhs + k_base / K_UNIT * K_UNIT * nr / RHS_ELEM_RECIP_BYTES;
            size_t n_offset = 0;

            for (; n_offset + 16 <= block_width; n_offset += 16) {
                uint8x16_t values[K_UNIT];
                uint8x16_t raw_sum = vdupq_n_u8(0);
                for (size_t k_offset = 0; k_offset < K_UNIT; ++k_offset) {
                    uint32_t input = 0xAAAAAAAAU;
                    if (k_base + k_offset < k) {
                        memcpy(
                            &input,
                            rhs_ptr + (k_base + k_offset) * rhs_stride +
                                (n_base + n_offset) / RHS_ELEM_RECIP_BYTES,
                            sizeof(input));
                    }
                    values[k_offset] = expand_qsu2(input);
                    raw_sum = vaddq_u8(raw_sum, values[k_offset]);
                }

                const uint8x16_t sign_conversion = vdupq_n_u8(0xAA);
                uint8x16x4_t packets;
                for (size_t byte = 0; byte < 4; ++byte) {
                    packets.val[byte] = veorq_u8(
                        vorrq_u8(
                            vorrq_u8(values[byte], vshlq_n_u8(values[4 + byte], 2)),
                            vorrq_u8(vshlq_n_u8(values[8 + byte], 4), vshlq_n_u8(values[12 + byte], 6))),
                        sign_conversion);
                }
                vst4q_u8(packed_unit + n_offset * 4, packets);
                accumulate_qsu2_sum(rhs_sums + n_offset, raw_sum);
            }

            for (; n_offset < block_width; ++n_offset) {
                const size_t n_idx = n_base + n_offset;
                for (size_t byte = 0; byte < 4; ++byte) {
                    uint8_t packed_byte = 0;
                    for (size_t field = 0; field < 4; ++field) {
                        const uint8_t qsu2 =
                            load_qsu2(rhs_ptr, rhs_stride, n_idx, k_base + byte + field * 4, n, k);
                        packed_byte |= (uint8_t)((qsu2 ^ RHS_ZERO_POINT) << (field * 2));
                        rhs_sums[n_offset] += (int32_t)qsu2 - RHS_ZERO_POINT;
                    }
                    packed_unit[n_offset * 4 + byte] = packed_byte;
                }
            }
        }

        size_t n_offset = 0;
        for (; n_offset + NR_TILE <= block_width; n_offset += NR_TILE) {
            const int32x4_t bias = vld1q_s32(bias_ptr + n_base + n_offset);
            const int32x4_t sum = vld1q_s32(rhs_sums + n_offset);
            vst1q_s32(packed_bias + n_offset, vmlaq_n_s32(bias, sum, k_sum_scale));

            const float32x4_t scale = vld1q_f32(scale_ptr + n_base + n_offset);
            vst1q_f32(packed_scale + n_offset, vmulq_n_f32(scale, scale_multiplier));
        }
        for (; n_offset < block_width; ++n_offset) {
            packed_bias[n_offset] = bias_ptr[n_base + n_offset] + k_sum_scale * rhs_sums[n_offset];
            packed_scale[n_offset] = scale_ptr[n_base + n_offset] * scale_multiplier;
        }
        for (; n_offset < nr; ++n_offset) {
            packed_bias[n_offset] = 0;
            packed_scale[n_offset] = 0.0F;
        }

        packed_ptr += packed_stride;
    }
}

struct kai_matmul_pack_rhs_uker_api
kai_matmul_pack_rhs_kxn_qsi2cxp16vsx4s12s8s4s0sf32bi32_qsu2cx_f32_i32_sme(void) {
    const struct kai_matmul_pack_rhs_uker_api api = {
        .run = run,
        .get_step = get_step,
        .get_rhs_stride = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,
        .get_rhs_packed_stride = get_rhs_packed_stride,
        .get_rhs_packed_offset = get_rhs_packed_offset,
        .get_rhs_packed_size = get_rhs_packed_size,
        .get_bias_n_offset = get_bias_n_offset,
        .get_scale_n_offset = get_scale_n_offset,
    };
    return api;
}

#endif  // Architectural features check.
