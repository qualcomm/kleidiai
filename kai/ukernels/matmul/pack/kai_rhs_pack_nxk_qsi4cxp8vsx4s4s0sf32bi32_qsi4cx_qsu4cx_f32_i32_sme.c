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

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs.h"
#include "kai/ukernels/matmul/kai_matmul_pack_rhs_types.h"

/// Public API for both source encodings of the "s4s0" nxk packer.
///
/// The destination format is signed either way ("qsi4cxp..."); the qsi4cx / qsu4cx suffix selects
/// how the SOURCE weights are encoded -- two's-complement nibbles, or offset-binary (value + 8).
/// The worker normalises the latter with an XOR, so both entry points emit identical packed bytes
/// and differ only in the zero point they pass. Every container query below is shared, which is
/// why the two entry points live in one file rather than two.
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

    MAX_NR = NR_VSCALE * KAI_VSCALE_MAX,
};

static size_t get_nr(void) {
    return NR_VSCALE * kai_get_sme_vscale();
}

static struct kai_matmul_pack_rhs_uker_dim_args kai_rhs_pack_nxk_s4s0_qsx4cx_get_step(
    const struct kai_matmul_pack_rhs_uker_config* config) {
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

static struct kai_matmul_pack_rhs_uker_rhs_stride_args kai_rhs_pack_nxk_s4s0_qsx4cx_get_rhs_stride(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_rhs_dim_args* shape) {
    KAI_UNUSED(config);
    const struct kai_matmul_pack_rhs_uker_rhs_stride_args stride = {
        .n = kai_roundup(shape->k, 2) / 2,
        .k = 0,
    };
    return stride;
}

static size_t kai_rhs_pack_nxk_s4s0_qsx4cx_get_rhs_offset(
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

static struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args kai_rhs_pack_nxk_s4s0_qsx4cx_get_rhs_packed_stride(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args stride = {
        .n = nr * (BIAS_ELEM_BYTES + (kai_roundup(shape->k, K_MULTIPLE) / RHS_ELEM_RECIP_BYTES) + SCALE_ELEM_BYTES),
    };
    return stride;
}

static size_t kai_rhs_pack_nxk_s4s0_qsx4cx_get_rhs_packed_offset(
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

static size_t kai_rhs_pack_nxk_s4s0_qsx4cx_get_rhs_packed_size(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_dim_args* shape,
    const struct kai_matmul_pack_rhs_uker_rhs_packed_stride_args* stride) {
    KAI_UNUSED(config);
    const size_t nr = get_nr();
    KAI_ASSUME(nr > 0);
    KAI_ASSUME(nr <= MAX_NR);
    return kai_roundup(shape->n, nr) / nr * stride->n;
}

static size_t kai_rhs_pack_nxk_s4s0_qsx4cx_get_bias_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_bias_n_dim_args* index) {
    KAI_UNUSED(config);
    return index->n * BIAS_ELEM_BYTES;
}

static size_t kai_rhs_pack_nxk_s4s0_qsx4cx_get_scale_n_offset(
    const struct kai_matmul_pack_rhs_uker_config* config,
    const struct kai_matmul_pack_rhs_uker_scale_n_dim_args* index) {
    KAI_UNUSED(config);
    return index->n * SCALE_ELEM_BYTES;
}

static void run_qsi4cx(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    kai_run_rhs_pack_nxk_qsi4cxp8vsx4s4s0sf32bi32_qsx4cx_f32_i32_sme(config, args, 0);
}

static void run_qsu4cx(
    const struct kai_matmul_pack_rhs_uker_config* config, const struct kai_matmul_pack_rhs_uker_args* args) {
    kai_run_rhs_pack_nxk_qsi4cxp8vsx4s4s0sf32bi32_qsx4cx_f32_i32_sme(config, args, 8);
}

static struct kai_matmul_pack_rhs_uker_api make_api(
    void (*run)(const struct kai_matmul_pack_rhs_uker_config*, const struct kai_matmul_pack_rhs_uker_args*)) {
    struct kai_matmul_pack_rhs_uker_api api = {
        .run = run,
        .get_step = kai_rhs_pack_nxk_s4s0_qsx4cx_get_step,
        .get_rhs_stride = kai_rhs_pack_nxk_s4s0_qsx4cx_get_rhs_stride,
        .get_rhs_offset = kai_rhs_pack_nxk_s4s0_qsx4cx_get_rhs_offset,
        .get_rhs_packed_stride = kai_rhs_pack_nxk_s4s0_qsx4cx_get_rhs_packed_stride,
        .get_rhs_packed_offset = kai_rhs_pack_nxk_s4s0_qsx4cx_get_rhs_packed_offset,
        .get_rhs_packed_size = kai_rhs_pack_nxk_s4s0_qsx4cx_get_rhs_packed_size,
        .get_bias_n_offset = kai_rhs_pack_nxk_s4s0_qsx4cx_get_bias_n_offset,
        .get_scale_n_offset = kai_rhs_pack_nxk_s4s0_qsx4cx_get_scale_n_offset,
    };
    return api;
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_nxk_qsi4cxp8vsx4s4s0sf32bi32_qsi4cx_f32_i32_sme(void) {
    return make_api(run_qsi4cx);
}

struct kai_matmul_pack_rhs_uker_api kai_matmul_pack_rhs_nxk_qsi4cxp8vsx4s4s0sf32bi32_qsu4cx_f32_i32_sme(void) {
    return make_api(run_qsu4cx);
}

#endif  // Architectural features check.
