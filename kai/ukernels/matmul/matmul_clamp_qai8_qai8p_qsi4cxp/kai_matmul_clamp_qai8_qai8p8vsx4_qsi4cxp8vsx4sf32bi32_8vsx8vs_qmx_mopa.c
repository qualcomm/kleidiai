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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

struct kai_matmul_uker_args_internal {
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
    void* accumulator_buffer;
    uint64_t flags;
    const void* lut_ptr;
};

enum {
    LHS_ELEM_BYTES = 1,
    RHS_ELEM_RECIP_BYTES = 2,
    DST_ELEM_BYTES = 1,
    BIAS_ELEM_BYTES = 4,
    SCALE_ELEM_BYTES = 4,

    MR_VSCALE = 8,
    NR_VSCALE = 8,
    KR = 4,
    K_MULTIPLE = 32,

    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

void kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_qmx_mopa(
    const struct kai_matmul_uker_args_internal* args);

// No LUT here: QMX has SME1 but not FEAT_LUT, so the i4 -> i8 expansion is arithmetic
// (lsl/asr + zip) inside the kernel. args.lut_ptr is left in the struct so the internal
// argument offsets stay identical to the sme2_mopa sibling.

static size_t get_mr(void) {
    return MR_VSCALE * kai_get_sme_vscale();
}

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

static size_t get_m_step(void) {
    return get_mr();
}

static size_t get_n_step(void) {
    return get_nr();
}

static struct kai_matmul_uker_dim_args get_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dim_args step = {
        .m = get_m_step(),
        .n = get_n_step(),
        .k = 0,
    };

    return step;
}

static struct kai_matmul_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_lhs_stride_args stride = {
        .m = get_mr() * kai_roundup(shape->k, KR) * LHS_ELEM_BYTES,
    };

    return stride;
}

static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_m_step() == 0);
    KAI_ASSUME(index->k == 0);

    return index->m / get_mr() * stride->m;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_rhs_stride_args stride = {
        .n = get_nr() *
            ((kai_roundup(shape->k, K_MULTIPLE) / RHS_ELEM_RECIP_BYTES + BIAS_ELEM_BYTES + SCALE_ELEM_BYTES)),
    };

    return stride;
}

static size_t get_rhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
    const struct kai_matmul_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->n % get_n_step() == 0);
    KAI_ASSUME(index->k == 0);

    return index->n / get_nr() * stride->n;
}

static struct kai_matmul_uker_dst_stride_args get_dst_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_dst_stride_args stride = {
        .m = shape->n * DST_ELEM_BYTES,
    };

    return stride;
}

static size_t get_dst_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m % get_m_step() == 0);
    KAI_ASSUME(index->n % get_n_step() == 0);

    return index->m * stride->m + index->n * DST_ELEM_BYTES;
}

static size_t get_dst_size(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);

    return shape->m * stride->m;
}

static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* args) {
    KAI_UNUSED(config);
    KAI_ASSUME_MSG((args->flags & ~((size_t)SUPPORTED_FLAGS)) == 0, "Only supported flags are accepted!");
    KAI_ASSUME(args->operand.lhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.dst.ptr != NULL);
    KAI_ASSUME(args->operand.bias.scale_bias_global.ptr != NULL);

    const bool enable_clamp = (args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;
    KAI_ASSUME(!enable_clamp || args->activation.clamp.min_ptr != NULL);
    KAI_ASSUME(!enable_clamp || args->activation.clamp.max_ptr != NULL);

    struct kai_matmul_uker_args_internal uker_args = {
        .A = args->operand.lhs.ptr,
        .B = args->operand.rhs.ptr,
        .C = args->operand.dst.ptr,
        .ldcb = args->operand.dst.stride.m,
        .M = args->shape.m,
        .N = args->shape.n,
        .K = args->shape.k,
        .min = enable_clamp ? *(const int32_t*)args->activation.clamp.min_ptr : INT8_MIN,
        .max = enable_clamp ? *(const int32_t*)args->activation.clamp.max_ptr : INT8_MAX,
        .result_zero_point = *(const int32_t*)args->operand.bias.scale_bias_global.ptr,
        .accumulator_buffer = NULL,
        .flags = 0,
        .lut_ptr = NULL,
    };

    kai_commit_za();

    kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_qmx_mopa(&uker_args);
}

struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_qmx_mopa(void) {
    struct kai_matmul_uker_api api = {
        .run = run,
        .get_step = get_step,
        .get_lhs_stride = get_lhs_stride,
        .get_lhs_offset = get_lhs_offset,
        .get_rhs_stride = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,
        .get_dst_stride = get_dst_stride,
        .get_dst_offset = get_dst_offset,
        .get_dst_size = get_dst_size,
    };

    return api;
}

#endif  // Architectural features check.
