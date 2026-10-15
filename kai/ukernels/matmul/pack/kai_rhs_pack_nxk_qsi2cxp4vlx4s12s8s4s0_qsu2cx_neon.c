//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "kai_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon.h"

#include <arm_neon.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);
static const size_t kai_k_multiple_of = 32;
static const size_t kai_num_rows_per_tile = 4;
static const size_t kai_num_k_units_per_tile = 4;

/// Regroups four 16-K units of packed QSU2 data and converts them to QSI2.
static inline uint32x4_t kai_repack_qsu2_packets4(const uint8x16_t src) {
    const uint32x4_t src_u32 = vreinterpretq_u32_u8(src);
    const uint32x4_t mask = vdupq_n_u32(0x03030303U);
    const uint32x4_t pack_multiplier = vdupq_n_u32(0x01041040U);

    const uint32x4_t field0 = vandq_u32(src_u32, mask);
    const uint32x4_t field1 = vandq_u32(vshrq_n_u32(src_u32, 2), mask);
    const uint32x4_t field2 = vandq_u32(vshrq_n_u32(src_u32, 4), mask);
    const uint32x4_t field3 = vandq_u32(vshrq_n_u32(src_u32, 6), mask);

    const uint32x4_t dst0 = vandq_u32(vshrq_n_u32(vmulq_u32(field0, pack_multiplier), 24), vdupq_n_u32(0x000000FFU));
    const uint32x4_t dst1 = vandq_u32(vshrq_n_u32(vmulq_u32(field1, pack_multiplier), 16), vdupq_n_u32(0x0000FF00U));
    const uint32x4_t dst2 = vandq_u32(vshrq_n_u32(vmulq_u32(field2, pack_multiplier), 8), vdupq_n_u32(0x00FF0000U));
    const uint32x4_t dst3 = vandq_u32(vmulq_u32(field3, pack_multiplier), vdupq_n_u32(0xFF000000U));

    const uint32x4_t packed = vorrq_u32(vorrq_u32(dst0, dst1), vorrq_u32(dst2, dst3));
    return veorq_u32(packed, vdupq_n_u32(0xAAAAAAAAU));
}

/// Regroups two 16-K units of packed QSU2 data and converts them to QSI2.
static inline uint32x2_t kai_repack_qsu2_packets2(const uint8x8_t src) {
    const uint8x16_t src_extended = vcombine_u8(src, vdup_n_u8(0));
    return vget_low_u32(kai_repack_qsu2_packets4(src_extended));
}

/// Calculates the signed row-sum contribution of 64 packed QSU2 values.
static inline int32_t kai_sum_qsu2_16_bytes(const uint8x16_t src) {
    const uint8x16_t mask = vdupq_n_u8(0x03);
    const uint8x16_t field0 = vandq_u8(src, mask);
    const uint8x16_t field1 = vandq_u8(vshrq_n_u8(src, 2), mask);
    const uint8x16_t field2 = vandq_u8(vshrq_n_u8(src, 4), mask);
    const uint8x16_t field3 = vshrq_n_u8(src, 6);
    const uint8x16_t raw_sum = vaddq_u8(vaddq_u8(field0, field1), vaddq_u8(field2, field3));

    return (int32_t)vaddlvq_u8(raw_sum) - 128;
}

/// Calculates the signed row-sum contribution of 32 packed QSU2 values.
static inline int32_t kai_sum_qsu2_8_bytes(const uint8x8_t src) {
    const uint8x8_t mask = vdup_n_u8(0x03);
    const uint8x8_t field0 = vand_u8(src, mask);
    const uint8x8_t field1 = vand_u8(vshr_n_u8(src, 2), mask);
    const uint8x8_t field2 = vand_u8(vshr_n_u8(src, 4), mask);
    const uint8x8_t field3 = vshr_n_u8(src, 6);
    const uint8x8_t raw_sum = vadd_u8(vadd_u8(field0, field1), vadd_u8(field2, field3));

    return (int32_t)vaddlv_u8(raw_sum) - 64;
}

/// Transposes and stores four rows containing four consecutive 16-K packets.
static inline void kai_store_rows4_k_units4(
    uint32_t* dst, const size_t nr, const uint32x4_t row0, const uint32x4_t row1, const uint32x4_t row2,
    const uint32x4_t row3) {
    const uint32x4_t rows01_lo = vzip1q_u32(row0, row1);
    const uint32x4_t rows01_hi = vzip2q_u32(row0, row1);
    const uint32x4_t rows23_lo = vzip1q_u32(row2, row3);
    const uint32x4_t rows23_hi = vzip2q_u32(row2, row3);

    const uint64x2_t units01 = vzip1q_u64(vreinterpretq_u64_u32(rows01_lo), vreinterpretq_u64_u32(rows23_lo));
    const uint64x2_t units23 = vzip2q_u64(vreinterpretq_u64_u32(rows01_hi), vreinterpretq_u64_u32(rows23_hi));

    vst1q_u32(dst, vreinterpretq_u32_u64(units01));
    vst1q_u32(dst + nr, vreinterpretq_u32_u64(vzip2q_u64(
                               vreinterpretq_u64_u32(rows01_lo), vreinterpretq_u64_u32(rows23_lo))));
    vst1q_u32(dst + (2 * nr), vreinterpretq_u32_u64(vzip1q_u64(
                                     vreinterpretq_u64_u32(rows01_hi), vreinterpretq_u64_u32(rows23_hi))));
    vst1q_u32(dst + (3 * nr), vreinterpretq_u32_u64(units23));
}

/// Transposes and stores four rows containing two consecutive 16-K packets.
static inline void kai_store_rows4_k_units2(
    uint32_t* dst, const size_t nr, const uint32x2_t row0, const uint32x2_t row1, const uint32x2_t row2,
    const uint32x2_t row3) {
    const uint32x2x2_t rows01 = vzip_u32(row0, row1);
    const uint32x2x2_t rows23 = vzip_u32(row2, row3);

    vst1q_u32(dst, vcombine_u32(rows01.val[0], rows23.val[0]));
    vst1q_u32(dst + nr, vcombine_u32(rows01.val[1], rows23.val[1]));
}

inline static size_t kai_k_roundedup(size_t k) {
    return kai_roundup(k, kai_k_multiple_of);
}

size_t kai_get_n_step_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(size_t nr) {
    return nr;
}

size_t kai_get_rhs_offset_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(size_t n_idx, size_t rhs_stride) {
    return n_idx * rhs_stride;
}

size_t kai_get_rhs_packed_stride_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(
    size_t k, size_t nr, size_t kr, size_t sr) {
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % 4) == 0);

    return nr * ((k_internal / 4) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias);
}

size_t kai_get_rhs_packed_offset_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(
    size_t n_idx, size_t k, size_t nr, size_t kr, size_t sr) {
    KAI_ASSERT((n_idx % nr) == 0);

    return (n_idx / nr) * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(k, nr, kr, sr);
}

size_t kai_get_rhs_packed_size_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(
    size_t n, size_t k, size_t nr, size_t kr, size_t sr) {
    const size_t num_rows = kai_roundup(n, nr) / nr;

    return num_rows * kai_get_rhs_packed_stride_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(k, nr, kr, sr);
}

void kai_run_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(
    size_t num_groups, size_t n, size_t k, size_t nr, size_t kr, size_t sr, const uint8_t* rhs, const float* bias,
    const float* scale, void* rhs_packed, size_t extra_bytes,
    const struct kai_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon_params* params, const int32_t* lut_arg) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % kr) == 0);
    KAI_ASSERT(num_groups == 1);
    KAI_ASSERT(extra_bytes == 0);
    KAI_ASSERT((kr % sr) == 0);
    KAI_ASSERT(rhs != NULL);
    KAI_ASSERT(scale != NULL);
    KAI_ASSERT(rhs_packed != NULL);
    KAI_ASSERT(params != NULL);
    KAI_ASSERT(params->lhs_zero_point == 1);
    KAI_ASSERT(params->rhs_zero_point == 2);
    KAI_ASSUME(k % kai_k_multiple_of == 0);
    KAI_ASSUME((kr / sr) == 4);
    KAI_ASSUME(nr > 0);
    KAI_ASSUME((nr % 4) == 0);
    KAI_UNUSED(lut_arg);

    const size_t rhs_stride = k_internal / 4;
    const size_t rhs_packed_stride =
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(k, nr, kr, sr);

    for (size_t row_idx = 0; row_idx < n; row_idx += nr) {
        uint8_t* const dst_panel = (uint8_t*)rhs_packed + ((row_idx / nr) * rhs_packed_stride);
        int32_t* const sums = (int32_t*)(dst_panel + (nr * (k_internal / 4)));
        float* const scaling_factors = (float*)((uint8_t*)sums + (nr * kai_num_bytes_sum_rhs));
        float* const biases = (float*)((uint8_t*)scaling_factors + (nr * kai_num_bytes_multiplier_rhs));
        const size_t rows_left = KAI_MIN(n - row_idx, nr);

        const size_t payload_size = nr * (k_internal / 4);
        memset(dst_panel, 0, payload_size);
        memset(sums, 0, nr * kai_num_bytes_sum_rhs);
        memcpy(scaling_factors, &scale[row_idx], rows_left * kai_num_bytes_multiplier_rhs);
        memset(&scaling_factors[rows_left], 0, (nr - rows_left) * kai_num_bytes_multiplier_rhs);

        if (bias == NULL) {
            memset(biases, 0, nr * kai_num_bytes_bias);
        } else {
            memcpy(biases, &bias[row_idx], rows_left * kai_num_bytes_bias);
            memset(&biases[rows_left], 0, (nr - rows_left) * kai_num_bytes_bias);
        }

        const size_t num_k_units = k_internal / 16;
        uint32_t* const dst_packets = (uint32_t*)dst_panel;
        size_t panel_row = 0;

        for (; panel_row + kai_num_rows_per_tile <= rows_left; panel_row += kai_num_rows_per_tile) {
            const uint8_t* const src_row0 = rhs + ((row_idx + panel_row) * rhs_stride);
            const uint8_t* const src_row1 = src_row0 + rhs_stride;
            const uint8_t* const src_row2 = src_row1 + rhs_stride;
            const uint8_t* const src_row3 = src_row2 + rhs_stride;
            int32_t row_sum0 = 0;
            int32_t row_sum1 = 0;
            int32_t row_sum2 = 0;
            int32_t row_sum3 = 0;
            size_t k_unit = 0;

            for (; k_unit + kai_num_k_units_per_tile <= num_k_units; k_unit += kai_num_k_units_per_tile) {
                const size_t src_offset = k_unit * 4;
                const uint8x16_t src0 = vld1q_u8(src_row0 + src_offset);
                const uint8x16_t src1 = vld1q_u8(src_row1 + src_offset);
                const uint8x16_t src2 = vld1q_u8(src_row2 + src_offset);
                const uint8x16_t src3 = vld1q_u8(src_row3 + src_offset);

                kai_store_rows4_k_units4(
                    dst_packets + (k_unit * nr) + panel_row, nr, kai_repack_qsu2_packets4(src0),
                    kai_repack_qsu2_packets4(src1), kai_repack_qsu2_packets4(src2),
                    kai_repack_qsu2_packets4(src3));

                row_sum0 += kai_sum_qsu2_16_bytes(src0);
                row_sum1 += kai_sum_qsu2_16_bytes(src1);
                row_sum2 += kai_sum_qsu2_16_bytes(src2);
                row_sum3 += kai_sum_qsu2_16_bytes(src3);
            }

            if (k_unit < num_k_units) {
                const size_t src_offset = k_unit * 4;
                const uint8x8_t src0 = vld1_u8(src_row0 + src_offset);
                const uint8x8_t src1 = vld1_u8(src_row1 + src_offset);
                const uint8x8_t src2 = vld1_u8(src_row2 + src_offset);
                const uint8x8_t src3 = vld1_u8(src_row3 + src_offset);

                KAI_ASSERT(k_unit + 2 == num_k_units);
                kai_store_rows4_k_units2(
                    dst_packets + (k_unit * nr) + panel_row, nr, kai_repack_qsu2_packets2(src0),
                    kai_repack_qsu2_packets2(src1), kai_repack_qsu2_packets2(src2),
                    kai_repack_qsu2_packets2(src3));

                row_sum0 += kai_sum_qsu2_8_bytes(src0);
                row_sum1 += kai_sum_qsu2_8_bytes(src1);
                row_sum2 += kai_sum_qsu2_8_bytes(src2);
                row_sum3 += kai_sum_qsu2_8_bytes(src3);
            }

            sums[panel_row] = row_sum0;
            sums[panel_row + 1] = row_sum1;
            sums[panel_row + 2] = row_sum2;
            sums[panel_row + 3] = row_sum3;
        }

        for (; panel_row < rows_left; ++panel_row) {
            const uint8_t* const src_row = rhs + ((row_idx + panel_row) * rhs_stride);
            int32_t row_sum = 0;
            size_t k_unit = 0;

            for (; k_unit + kai_num_k_units_per_tile <= num_k_units; k_unit += kai_num_k_units_per_tile) {
                const uint8x16_t src = vld1q_u8(src_row + (k_unit * 4));
                const uint32x4_t packed = kai_repack_qsu2_packets4(src);

                dst_packets[(k_unit * nr) + panel_row] = vgetq_lane_u32(packed, 0);
                dst_packets[((k_unit + 1) * nr) + panel_row] = vgetq_lane_u32(packed, 1);
                dst_packets[((k_unit + 2) * nr) + panel_row] = vgetq_lane_u32(packed, 2);
                dst_packets[((k_unit + 3) * nr) + panel_row] = vgetq_lane_u32(packed, 3);
                row_sum += kai_sum_qsu2_16_bytes(src);
            }

            if (k_unit < num_k_units) {
                const uint8x8_t src = vld1_u8(src_row + (k_unit * 4));
                const uint32x2_t packed = kai_repack_qsu2_packets2(src);

                KAI_ASSERT(k_unit + 2 == num_k_units);
                dst_packets[(k_unit * nr) + panel_row] = vget_lane_u32(packed, 0);
                dst_packets[((k_unit + 1) * nr) + panel_row] = vget_lane_u32(packed, 1);
                row_sum += kai_sum_qsu2_8_bytes(src);
            }

            sums[panel_row] = row_sum;
        }
    }
}
#endif  // Architectural features check.
