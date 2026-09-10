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
    int32_t c_offset;
    int32_t maxval;
    int32_t minval;
    const void* A_ptr;
    const void* B_ptr;
    size_t N;
    size_t K;
    void* output_ptr;
    uint64_t flags;
    const void* lut_ptr;
};

enum {
    LHS_ELEM_BYTES = 1,
    RHS_ELEM_RECIP_BYTES = 2,
    DST_ELEM_BYTES = 1,
    BIAS_ELEM_BYTES = 4,
    SCALE_ELEM_BYTES = 4,

    MR = 1,
    NR_VSCALE = 8,
    // One nr panel per outer iteration, and n_step == nr.
    //
    // A four-panel internal unroll was built and measured. It amortises the shared LHS ld1rqb and
    // the loop overhead over four panels, cutting the K loop from 51 instructions per 1024 MACs to
    // 174 per 4096 (-16%); SME1 has no multi-vector SDOT, so the SDOT-per-MAC ratio is fixed and
    // this is the only saving available. Measured against sme2_dot on the same run, it was worth
    // +20% where the packed RHS fits in cache (512^2, 1024^2, 4096^2) but LOST up to 19% on the
    // large streaming shapes (8192^2, and the 4096/11008 pair), because four panels mean four
    // concurrent RHS streams ~64KB apart instead of one sequential walk, and a GEMV over i4 weights
    // is prefetch-bound there. The streaming shapes are the ones that matter for LLM FFN GEMV, and
    // the unrolled version cost 340 extra lines of asm plus a second code path, so it was dropped.
    //
    // Note n_step is the granularity a CALLER may tile N at, not an unroll width:
    // get_rhs_offset/get_dst_offset assume index->n is a multiple of it, so raising it to 4*nr
    // forbids every caller that slices N at nr -- which the test does, and which aborts on the
    // assume. Any future unroll belongs entirely inside the asm, with n_step left alone.
    N_STEP_VSCALE = NR_VSCALE,
    KR = 4,
    K_MULTIPLE = 32,

    SUPPORTED_FLAGS = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
};

void kai_kernel_matmul_clamp_qai8_qai8_qsi4cxp8vsx4s4s0sf32bi32_1x8vs_qmx_dot(
    const struct kai_matmul_uker_args_internal* args_ptr);

static size_t get_mr(void) {
    return MR;
}

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

static size_t get_m_step(void) {
    return get_mr();
}

static size_t get_n_step(void) {
    return N_STEP_VSCALE * kai_get_sme_vscale();
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
        .m = shape->k * LHS_ELEM_BYTES,
    };

    return stride;
}

static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->m == 0);
    KAI_ASSUME(index->k == 0);

    return index->m * stride->m;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);

    const struct kai_matmul_uker_rhs_stride_args stride = {
        .n = get_nr() * (kai_roundup(shape->k, K_MULTIPLE) / RHS_ELEM_RECIP_BYTES + BIAS_ELEM_BYTES + SCALE_ELEM_BYTES),
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
    KAI_ASSUME(index->m == 0);
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
    KAI_ASSUME(args->shape.m == 1);
    KAI_ASSUME(args->operand.lhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.dst.ptr != NULL);
    KAI_ASSUME(args->operand.bias.scale_bias_global.ptr != NULL);

    const bool enable_clamp = (args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;
    KAI_ASSUME(!enable_clamp || args->activation.clamp.min_ptr != NULL);
    KAI_ASSUME(!enable_clamp || args->activation.clamp.max_ptr != NULL);

    struct kai_matmul_uker_args_internal uker_args = {
        .c_offset = *(const int32_t*)args->operand.bias.scale_bias_global.ptr,
        .maxval = enable_clamp ? *(const int32_t*)args->activation.clamp.max_ptr : INT8_MAX,
        .minval = enable_clamp ? *(const int32_t*)args->activation.clamp.min_ptr : INT8_MIN,
        .A_ptr = args->operand.lhs.ptr,
        .B_ptr = args->operand.rhs.ptr,
        .N = args->shape.n,
        .K = args->shape.k,
        .output_ptr = args->operand.dst.ptr,
        .flags = 2,
        // No LUT: the i4 -> i8 expansion is arithmetic (lsl #4 / and #0xf0), because SME1 has no
        // FEAT_LUT. The field is kept so every other offset matches the sme2_dot sibling's struct.
        .lut_ptr = NULL,
    };

    kai_commit_za();

    kai_kernel_matmul_clamp_qai8_qai8_qsi4cxp8vsx4s4s0sf32bi32_1x8vs_qmx_dot(&uker_args);
}

struct kai_matmul_uker_api kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4s4s0sf32bi32_1x8vs_qmx_dot(void) {
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
