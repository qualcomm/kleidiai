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
    K_UNITS_PER_TILE = 4,

    RHS_ZERO_POINT = 2,
    MAX_NR = NR_VSCALE * KAI_VSCALE_MAX,
};

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

/// Regroups four consecutive 16-K units and converts offset-binary QSU2 to two's-complement QSI2.
static inline uint32x4_t repack_qsu2_k_units4(const uint8x16_t src) {
    const uint32x4_t src_u32 = vreinterpretq_u32_u8(src);
    const uint32x4_t mask = vdupq_n_u32(0x03030303U);
    const uint32x4_t pack_multiplier = vdupq_n_u32(0x01041040U);

    const uint32x4_t field0 = vandq_u32(src_u32, mask);
    const uint32x4_t field1 = vandq_u32(vshrq_n_u32(src_u32, 2), mask);
    const uint32x4_t field2 = vandq_u32(vshrq_n_u32(src_u32, 4), mask);
    const uint32x4_t field3 = vandq_u32(vshrq_n_u32(src_u32, 6), mask);

    const uint32x4_t dst0 =
        vandq_u32(vshrq_n_u32(vmulq_u32(field0, pack_multiplier), 24), vdupq_n_u32(0x000000FFU));
    const uint32x4_t dst1 =
        vandq_u32(vshrq_n_u32(vmulq_u32(field1, pack_multiplier), 16), vdupq_n_u32(0x0000FF00U));
    const uint32x4_t dst2 =
        vandq_u32(vshrq_n_u32(vmulq_u32(field2, pack_multiplier), 8), vdupq_n_u32(0x00FF0000U));
    const uint32x4_t dst3 = vandq_u32(vmulq_u32(field3, pack_multiplier), vdupq_n_u32(0xFF000000U));

    const uint32x4_t packed = vorrq_u32(vorrq_u32(dst0, dst1), vorrq_u32(dst2, dst3));
    return veorq_u32(packed, vdupq_n_u32(0xAAAAAAAAU));
}

/// Signed row-sum contribution of 64 QSU2 values.
static inline int32_t sum_qsu2_16_bytes(const uint8x16_t src) {
    const uint8x16_t mask = vdupq_n_u8(0x03);
    const uint8x16_t field0 = vandq_u8(src, mask);
    const uint8x16_t field1 = vandq_u8(vshrq_n_u8(src, 2), mask);
    const uint8x16_t field2 = vandq_u8(vshrq_n_u8(src, 4), mask);
    const uint8x16_t field3 = vshrq_n_u8(src, 6);
    const uint8x16_t raw_sum = vaddq_u8(vaddq_u8(field0, field1), vaddq_u8(field2, field3));
    return (int32_t)vaddlvq_u8(raw_sum) - 2 * K_UNIT * K_UNITS_PER_TILE;
}

static inline uint8_t get_qsu2(const uint8_t* row, const size_t k_idx) {
    return (row[k_idx / RHS_ELEM_RECIP_BYTES] >> (2 * (k_idx % RHS_ELEM_RECIP_BYTES))) & 0x03;
}

/// Packs one 16-K unit as four bytes. Byte r contains K+r, K+4+r, K+8+r, K+12+r.
static inline uint32_t repack_qsu2_k_unit(const uint8_t* row, const size_t k_base, const size_t k, int32_t* sum) {
    uint32_t packet = 0;
    for (size_t r = 0; r < 4; ++r) {
        uint8_t packed_byte = 0;
        for (size_t field = 0; field < 4; ++field) {
            const size_t k_idx = k_base + r + field * 4;
            const uint8_t qsu2 = k_idx < k ? get_qsu2(row, k_idx) : RHS_ZERO_POINT;
            packed_byte |= (uint8_t)((qsu2 ^ RHS_ZERO_POINT) << (2 * field));
            if (k_idx < k) {
                *sum += (int32_t)qsu2 - RHS_ZERO_POINT;
            }
        }
        packet |= (uint32_t)packed_byte << (8 * r);
    }
    return packet;
}

static struct kai_matmul_pack_rhs_uker_dim_args get_step(const struct kai_matmul_pack_rhs_uker_config* config) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
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
        .n = kai_roundup(shape->k, RHS_ELEM_RECIP_BYTES) / RHS_ELEM_RECIP_BYTES,
        .k = 0,
    };
    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* index,
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);
    KAI_ASSUME(index->n % nr == 0);
    KAI_ASSUME(index->k == 0);
    return index->n * stride->n;
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

/// Packs signed s12s8s4s0 packets from an NxK QSU2 matrix.
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
    const size_t num_k_units = rounded_k / K_UNIT;
    const size_t rhs_stride = args->operand.rhs.stride.n;
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

        for (size_t n_idx = 0; n_idx < block_width; ++n_idx) {
            const uint8_t* row = rhs_ptr + (n_base + n_idx) * rhs_stride;
            size_t k_unit = 0;
            int32_t sum = 0;

            for (; k_unit + K_UNITS_PER_TILE <= num_k_units && (k_unit + K_UNITS_PER_TILE) * K_UNIT <= k;
                 k_unit += K_UNITS_PER_TILE) {
                const uint8x16_t src = vld1q_u8(row + k_unit * K_UNIT / RHS_ELEM_RECIP_BYTES);
                const uint32x4_t packets = repack_qsu2_k_units4(src);
                uint32_t* const unit0 = (uint32_t*)(packed_rhs + (k_unit + 0) * 4 * nr);
                uint32_t* const unit1 = (uint32_t*)(packed_rhs + (k_unit + 1) * 4 * nr);
                uint32_t* const unit2 = (uint32_t*)(packed_rhs + (k_unit + 2) * 4 * nr);
                uint32_t* const unit3 = (uint32_t*)(packed_rhs + (k_unit + 3) * 4 * nr);
                unit0[n_idx] = vgetq_lane_u32(packets, 0);
                unit1[n_idx] = vgetq_lane_u32(packets, 1);
                unit2[n_idx] = vgetq_lane_u32(packets, 2);
                unit3[n_idx] = vgetq_lane_u32(packets, 3);
                sum += sum_qsu2_16_bytes(src);
            }

            for (; k_unit < num_k_units; ++k_unit) {
                const uint32_t packet = repack_qsu2_k_unit(row, k_unit * K_UNIT, k, &sum);
                memcpy(packed_rhs + k_unit * 4 * nr + n_idx * 4, &packet, sizeof(packet));
            }
            rhs_sums[n_idx] = sum;
        }

        size_t n_idx = 0;
        for (; n_idx + NR_TILE <= block_width; n_idx += NR_TILE) {
            const int32x4_t bias = vld1q_s32(bias_ptr + n_base + n_idx);
            const int32x4_t sum = vld1q_s32(rhs_sums + n_idx);
            vst1q_s32(packed_bias + n_idx, vmlaq_n_s32(bias, sum, k_sum_scale));

            const float32x4_t scale = vld1q_f32(scale_ptr + n_base + n_idx);
            vst1q_f32(packed_scale + n_idx, vmulq_n_f32(scale, scale_multiplier));
        }
        for (; n_idx < block_width; ++n_idx) {
            packed_bias[n_idx] = bias_ptr[n_base + n_idx] + k_sum_scale * rhs_sums[n_idx];
            packed_scale[n_idx] = scale_ptr[n_base + n_idx] * scale_multiplier;
        }
        for (; n_idx < nr; ++n_idx) {
            packed_bias[n_idx] = 0;
            packed_scale[n_idx] = 0.0F;
        }

        packed_ptr += packed_stride;
    }
}

struct kai_matmul_pack_rhs_uker_api
kai_matmul_pack_rhs_nxk_qsi2cxp16vsx4s12s8s4s0sf32bi32_qsu2cx_f32_i32_sme(void) {
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
