//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//

#if (!defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2) || !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)) && \
    !defined(_M_ARM64)
#error This file must be compiled for AArch64, FEAT_SVE2 and FEAT_FP16.
#else  // Architectural features check.

#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"

typedef struct {
    float* dst;                  // 0x00
    const void* lhs_packed;      // 0x08
    const void* rhs_packed;      // 0x10
    const uint16_t* rhs_scales;  // 0x18
    size_t dst_stride_row;       // 0x20
    size_t lhs_packed_stride;    // 0x28
    size_t rhs_packed_stride;    // 0x30
    size_t m;                    // 0x38
    size_t n;                    // 0x40
    size_t k;                    // 0x48
    size_t bl;                   // 0x50
    float scalar_min;            // 0x58
    float scalar_max;            // 0x5c
} KernelArgs;

void kai_kernel_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa(KernelArgs* args_ptr);

// Compute args
static const size_t kai_m_step = 4;   // Multiple of vector scale
static const size_t kai_n_step = 16;  // Multiple of vector scale
// Packing args
static const size_t kai_mr = 4;   // Multiple of vector scale
static const size_t kai_nr = 16;  // Multiple of vector scale
// LHS format args (num. bytes per value, multiplier, zero_point (if asymmetric))
static const size_t kai_num_bytes_qvalue_lhs = 2;
// RHS format args (num. bytes per value, multiplier, zero_point (if asymmetric), and reduction sum (if LHS is
// asymmetric))
static const size_t kai_recip_num_bytes_qvalue_rhs = 2;
static const size_t kai_num_bytes_multiplier_rhs = 2;
// Extra args
static const size_t kai_bl = 32;

static size_t kai_get_mr_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa(void);
static size_t kai_get_nr_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa(void);

inline static size_t kai_get_num_bytes_per_block_lhs(size_t bl) {
    return bl * kai_num_bytes_qvalue_lhs;
}

inline static size_t kai_get_num_bytes_per_block_rhs(size_t bl) {
    KAI_ASSUME(bl != 0);
    KAI_ASSUME((bl % kai_bl) == 0);
    return (bl / kai_recip_num_bytes_qvalue_rhs) + kai_num_bytes_multiplier_rhs;
}

inline static size_t kai_get_num_blocks_per_row(size_t k, size_t bl) {
    KAI_ASSUME(bl != 0);
    KAI_ASSUME((bl % kai_bl) == 0);
    KAI_ASSUME((k % bl) == 0);
    return k / bl;
}

inline static size_t kai_get_lhs_packed_stride(size_t k, size_t bl) {
    const size_t mr = kai_get_mr_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa();
    return mr * kai_get_num_blocks_per_row(k, bl) * kai_get_num_bytes_per_block_lhs(bl);
}

inline static size_t kai_get_rhs_packed_stride(size_t k, size_t bl) {
    KAI_ASSUME(bl != 0);
    KAI_ASSUME((bl % kai_bl) == 0);
    KAI_ASSUME((k % bl) == 0);

    const size_t num_blocks_per_row = kai_get_num_blocks_per_row(k, bl);
    const size_t num_bytes_per_block = kai_get_num_bytes_per_block_rhs(bl);
    const size_t nr = kai_get_nr_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa();
    return nr * num_bytes_per_block * num_blocks_per_row;
}

static size_t kai_get_m_step_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa(void) {
    return kai_m_step * kai_get_sme_vscale();
}

static size_t kai_get_n_step_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa(void) {
    return kai_n_step * kai_get_sme_vscale();
}

static size_t kai_get_mr_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa(void) {
    return kai_mr * kai_get_sme_vscale();
}

static size_t kai_get_nr_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa(void) {
    return kai_nr * kai_get_sme_vscale();
}

static struct kai_matmul_uker_dim_args get_step(const struct kai_matmul_uker_config* config) {
    KAI_UNUSED(config);
    return (struct kai_matmul_uker_dim_args){
        .m = kai_get_m_step_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa(),
        .n = kai_get_n_step_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa(),
        .k = 0};
}

static struct kai_matmul_uker_lhs_stride_args get_lhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* shape) {
    return (struct kai_matmul_uker_lhs_stride_args){.m = kai_get_lhs_packed_stride(shape->k, config->format.bl)};
}

static size_t get_lhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_lhs_dim_args* index,
    const struct kai_matmul_uker_lhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->k == 0);
    return index->m / kai_get_mr_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa() * stride->m;
}

static struct kai_matmul_uker_rhs_stride_args get_rhs_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* shape) {
    return (struct kai_matmul_uker_rhs_stride_args){.n = kai_get_rhs_packed_stride(shape->k, config->format.bl)};
}

static size_t get_rhs_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_rhs_dim_args* index,
    const struct kai_matmul_uker_rhs_stride_args* stride) {
    KAI_UNUSED(config);
    KAI_ASSUME(index->k == 0);
    return index->n / kai_get_nr_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa() * stride->n;
}

static struct kai_matmul_uker_dst_stride_args get_dst_stride(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape) {
    KAI_UNUSED(config);
    return (struct kai_matmul_uker_dst_stride_args){.m = shape->n * sizeof(float)};
}

static size_t get_dst_offset(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* index,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    return index->m * stride->m + index->n * sizeof(float);
}

static size_t get_dst_size(
    const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_dst_dim_args* shape,
    const struct kai_matmul_uker_dst_stride_args* stride) {
    KAI_UNUSED(config);
    return shape->m * stride->m;
}

static void run(const struct kai_matmul_uker_config* config, const struct kai_matmul_uker_args* args) {
    KAI_ASSUME(config != NULL);
    KAI_ASSUME(args != NULL);
    KAI_ASSUME(config->format.bl != 0);
    KAI_ASSUME((config->format.bl % kai_bl) == 0);
    KAI_ASSUME(args->shape.k != 0);
    KAI_ASSUME(args->operand.lhs.ptr != NULL);
    KAI_ASSUME(args->operand.rhs.ptr != NULL);
    KAI_ASSUME(args->operand.dst.ptr != NULL);
    KAI_ASSUME((args->flags & ~((uint64_t)KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP)) == 0);
    const bool clamp = (args->flags & KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP) != 0;
    KAI_ASSUME(!clamp || args->activation.clamp.min_ptr != NULL);
    KAI_ASSUME(!clamp || args->activation.clamp.max_ptr != NULL);
    const float min = clamp ? *(const float*)args->activation.clamp.min_ptr : -FLT_MAX;
    const float max = clamp ? *(const float*)args->activation.clamp.max_ptr : FLT_MAX;

    if (args->shape.m == 0 || args->shape.n == 0) {
        return;
    }

    const size_t bl = config->format.bl;
    const size_t num_blocks = kai_get_num_blocks_per_row(args->shape.k, bl);
    const size_t nr = kai_get_nr_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa();
    const size_t rhs_packed_stride = kai_get_rhs_packed_stride(args->shape.k, bl);
    const uint16_t* rhs_scales = (const uint16_t*)((const uint8_t*)args->operand.rhs.ptr + rhs_packed_stride -
                                                   (nr * num_blocks) * kai_num_bytes_multiplier_rhs);

    KernelArgs kernel_args = {
        .dst = (float*)args->operand.dst.ptr,
        .lhs_packed = args->operand.lhs.ptr,
        .rhs_packed = args->operand.rhs.ptr,
        .rhs_scales = rhs_scales,
        .dst_stride_row = args->operand.dst.stride.m,
        .lhs_packed_stride = kai_get_lhs_packed_stride(args->shape.k, bl),
        .rhs_packed_stride = rhs_packed_stride,
        .m = args->shape.m,
        .n = args->shape.n,
        .k = args->shape.k,
        .bl = bl,
        .scalar_min = min,
        .scalar_max = max,
    };

    kai_commit_za();
    kai_kernel_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa(&kernel_args);
}

struct kai_matmul_uker_api kai_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa(void) {
    return (struct kai_matmul_uker_api){
        .run = run,
        .get_step = get_step,
        .get_lhs_stride = get_lhs_stride,
        .get_lhs_offset = get_lhs_offset,
        .get_rhs_stride = get_rhs_stride,
        .get_rhs_offset = get_rhs_offset,
        .get_dst_stride = get_dst_stride,
        .get_dst_offset = get_dst_offset,
        .get_dst_size = get_dst_size};
}

#endif  // Architectural features check.
