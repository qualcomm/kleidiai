//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)) && !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.
#include "kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

typedef struct {
    const void* A;
    const void* B;
    void* C;
    uint64_t ldcb;
    uint64_t M;
    uint64_t N;
    uint64_t K;
    int32_t min;
    int32_t max;
    int32_t result_zero_point;
    const int n_0;
    void* accumulator_buffer;
    uint64_t flags;
} KernelArgs;

static const size_t kai_mr = 2;
static const size_t kai_nr = 2;
static const size_t kai_kr = 4;
static const size_t kai_sr = 1;

void kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(KernelArgs* args);

// Returns a constant value specific to this kernel that's relative to vector length
static size_t kai_get_kernel_vec_length_constant(void) {
    const size_t kernel_vec_length_constant = kai_get_sme_vector_length_u8() / kai_kr;
    return kernel_vec_length_constant;
}

size_t kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void) {
    return kai_mr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void) {
    return kai_nr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void) {
    return kai_mr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void) {
    return kai_nr * kai_get_kernel_vec_length_constant();
}

size_t kai_get_kr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(size_t m_idx, size_t k) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa() == 0);
    return m_idx * kai_roundup(k, kai_kr) * sizeof(int8_t);
}

static size_t kai_get_rhs_packed_stride_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(size_t k) {
    return kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa() *
        (sizeof(int32_t) + kai_roundup(k, kai_kr) * sizeof(int8_t) + sizeof(float));
}

size_t kai_get_rhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa() == 0);
    const size_t block_idx = n_idx / kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
    return block_idx * kai_get_rhs_packed_stride_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(k);
}

size_t kai_get_dst_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride_row) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa() == 0);

    return m_idx * dst_stride_row + n_idx * sizeof(int8_t);
}

size_t kai_get_dst_size_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(size_t m, size_t n) {
    return m * n * sizeof(int8_t);
}

void kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct kai_matmul_requantize32_params* params) {
    KAI_ASSUME(dst_stride_col == sizeof(int8_t));
    KernelArgs args;

    args.A = lhs_packed;
    args.B = rhs_packed;
    args.C = dst;
    args.ldcb = dst_stride_row;
    args.M = m;
    args.N = n;
    args.K = k;
    args.min = params->min_value;
    args.max = params->max_value;
    args.result_zero_point = params->output_zero_point;
    args.accumulator_buffer = NULL;
    args.flags = 0;

    kai_kernel_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(&args);
}

#endif  // Architectural features check.
