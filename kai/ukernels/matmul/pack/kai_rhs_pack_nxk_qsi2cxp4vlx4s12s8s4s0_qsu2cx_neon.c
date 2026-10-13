//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__aarch64__) && !defined(_M_ARM64)
#error This file must be compiled for AArch64.
#else  // Architectural features check.

#include "kai_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "kai/kai_common.h"

static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_bias = sizeof(float);
static const size_t kai_k_multiple_of = 32;

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

        memset(sums, 0, nr * kai_num_bytes_sum_rhs);
        memcpy(scaling_factors, &scale[row_idx], rows_left * kai_num_bytes_multiplier_rhs);
        memset(&scaling_factors[rows_left], 0, (nr - rows_left) * kai_num_bytes_multiplier_rhs);

        if (bias == NULL) {
            memset(biases, 0, nr * kai_num_bytes_bias);
        } else {
            memcpy(biases, &bias[row_idx], rows_left * kai_num_bytes_bias);
            memset(&biases[rows_left], 0, (nr - rows_left) * kai_num_bytes_bias);
        }

        for (size_t k_unit = 0; k_unit < k_internal / 16; ++k_unit) {
            uint8_t* const dst_k_unit = dst_panel + (k_unit * 4 * nr);

            for (size_t panel_row = 0; panel_row < nr; ++panel_row) {
                const bool is_valid_row = panel_row < rows_left;
                const uint8_t* src_row = NULL;
                if (is_valid_row) {
                    src_row = rhs + ((row_idx + panel_row) * rhs_stride);
                }

                for (size_t k_in_group = 0; k_in_group < 4; ++k_in_group) {
                    uint8_t dst_byte = 0;

                    for (size_t field = 0; field < 4; ++field) {
                        const size_t k_idx = (k_unit * 16) + (field * 4) + k_in_group;
                        uint8_t raw_value = 2;
                        if (is_valid_row) {
                            raw_value = (src_row[k_idx / 4] >> ((k_idx % 4) * 2)) & 0x03;
                        }
                        const uint8_t signed_code = raw_value ^ 0x02;
                        dst_byte |= signed_code << (field * 2);

                        if (is_valid_row) {
                            sums[panel_row] += (int32_t)raw_value - 2;
                        }
                    }

                    dst_k_unit[(panel_row * 4) + k_in_group] = dst_byte;
                }
            }
        }
    }
}
#endif  // Architectural features check.
