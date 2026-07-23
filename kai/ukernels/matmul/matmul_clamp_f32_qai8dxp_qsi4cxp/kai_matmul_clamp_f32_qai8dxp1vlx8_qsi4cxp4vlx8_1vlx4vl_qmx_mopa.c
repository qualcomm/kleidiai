//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//


// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural feature check

#include "kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 1;  // multiple of vector length
static const size_t kai_nr = 4;  // multiple of vector length
static const size_t kai_kr = 8;
static const size_t kai_sr = 2;
static const size_t kai_num_bytes_multiplier_lhs = sizeof(float);
static const size_t kai_num_bytes_multiplier_rhs = sizeof(float);
static const size_t kai_num_bytes_offset_lhs = sizeof(int32_t);
static const size_t kai_num_bytes_sum_rhs = sizeof(int32_t);
static const size_t kai_num_bytes_bias_rhs = sizeof(float);
static const size_t kai_k_multiple_of = 32;

inline static size_t kai_k_roundedup(size_t k) {
    // Round up k to be a multiple of 32.
    return kai_roundup(k, kai_k_multiple_of);
}

inline static size_t kai_get_lhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % kai_k_multiple_of) == 0);

    return kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa() *
        (k_internal * sizeof(int8_t) + kai_num_bytes_multiplier_lhs + kai_num_bytes_offset_lhs);
}

inline static size_t kai_get_rhs_packed_stride(size_t k) {
    const size_t k_internal = kai_k_roundedup(k);

    KAI_ASSERT((k_internal % kai_k_multiple_of) == 0);

    return kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa() *
        ((k_internal / 2) + kai_num_bytes_multiplier_rhs + kai_num_bytes_sum_rhs + kai_num_bytes_bias_rhs);
}

size_t kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa(size_t m_idx, size_t k) {
    KAI_ASSERT((m_idx % kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa()) == 0);

    const size_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa();

    return (m_idx / mr) * kai_get_lhs_packed_stride(k);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa(size_t n_idx, size_t k) {
    KAI_ASSERT((n_idx % kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa()) == 0);

    const size_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa();

    return (n_idx / nr) * kai_get_rhs_packed_stride(k);
}

size_t kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSERT((m_idx % kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa()) == 0);
    KAI_ASSERT((n_idx % kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa()) == 0);

    return (n_idx * sizeof(float) + m_idx * dst_stride);
}

size_t kai_get_dst_size_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa(
    size_t m, size_t n, size_t k, const void* restrict lhs_packed, const void* restrict rhs_packed,
    float* restrict dst,  // NOLINT(readability-non-const-parameter)
    size_t dst_stride_row, size_t dst_stride_col, float scalar_min, float scalar_max) {
    KAI_ASSERT(dst_stride_col == sizeof(float));
    KAI_ASSERT(n > 0);
    KAI_ASSERT(m > 0);

    // Constants
    uint64_t mr = kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa();
    uint64_t nr = kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa();
    uint64_t lhs_stride = kai_get_lhs_packed_stride(k);
    uint64_t rhs_stride = kai_get_rhs_packed_stride(k);
    uint64_t k_internal = kai_k_roundedup(k);
    uint64_t m_blk = k_internal * mr;
    uint64_t dst_inc = mr * dst_stride_row;
    uint64_t rhs_row_bytes = nr * (k_internal / 2);
    float scalar_bounds[2] = {scalar_min, scalar_max};

    /* ---------------------------------------------------
                  Registers allocations
        x7:  SME vector length in words (cntw)
        x8:  RHS base address (rhs)
        x9:  Destination base address (dst)
        x10: LHS pointer (lhs)
        x11: RHS pointer (rhs)
        x12: Remaining M elements
        x13: Remaining N elements
        x14: k exit condition (k_cond)
             ZA tile index (l_idx)
        x15: LHS scaling factor pointer (lhs_sf_ptr)
        x16: ZA tile exit condition (l_cnd)
        x17: Destination pointer (dst)
        x19: Destination outer address (dst)
        x20: LHS base address (lhs)
        x28: RHS ragged-tail byte boundary
        z28: 0xf0 mask, used to isolate the high nibble (already pre-scaled by 16 in its byte position)
    --------------------------------------------------- */
    kai_commit_za();
    __asm__ volatile(
        "   .inst 0xd503477f // smstart                       \n"
        "   mov   x19, %[dst]                                \n"
        "   mov   x20, %[lhs]                                \n"
        "   cntw  x7                                         \n"
        "   mov   w9, #0xf0                                  \n"
        "   dup   z28.b, w9                                  \n"
        "   ptrue p2.b                                       \n"
        "   ld1rw {z30.s}, p2/Z, [%[scalar_bounds]]          \n"
        "   ld1rw {z31.s}, p2/Z, [%[scalar_bounds], #4]      \n"
        "   mov     x12, %[m]                                \n"
        "   whilelt p0.s, xzr, x12                           \n"
        "1:                                                  \n"
        "   mov     x8, %[rhs]                               \n"
        "   add     x28, x8, %[rhs_row_bytes]                \n"
        "   mov     x9, x19                                  \n"
        "   mov     x13, %[n]                                \n"
        "   cmp     x7, x12                                  \n"
        "   csel    x16, x7, x12, lt                         \n"
        "   lsl     x16, x16, #2                             \n"
        "	mov x21, #0      								 \n"
        "2:                                                  \n"
        "   mov     x10, x20                                 \n"
        "   mov     x11, x8                                  \n"
        "   mov     x17, x9                                  \n"
        "   .inst 0xc00800ff // zero    {za}                  \n"
        "   add     x14, x10, %[m_blk]                       \n"
        "3:                                                  \n"
        // LHS: two groups of kai_kr=4 int8 K-values, 16 elements (one VL of int8 packed as int32) apart,
        // matching the RHS nibble-pairing distance produced by kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.
        "   ld1w    { z16.s }, p0/z, [x10]                   \n"
        "   ld1w    { z20.s }, p0/z, [x10, #0x4, mul vl]     \n"
        // RHS: each of the kai_nr=4 output lanes owns its own separate byte stream (no two lanes share a
        // loaded register), matching the per-lane layout produced by kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.
        "   mov     x15, x11                                 \n"
        "   whilelt p3.h, x15, x28                           \n"
        "   addvl   x15, x11, #1                             \n"
        "   whilelt p4.h, x15, x28                           \n"
        "   addvl   x15, x11, #2                             \n"
        "   whilelt p6.h, x15, x28                           \n"
        "   addvl   x15, x11, #3                             \n"
        "   whilelt p7.h, x15, x28                           \n"
        "   ld1h    z0.h, p3/z, [x11]                        \n"
        "   ld1h    z1.h, p4/z, [x11, #1, mul vl]             \n"
        "   ld1h    z2.h, p6/z, [x11, #2, mul vl]             \n"
        "   ld1h    z3.h, p7/z, [x11, #3, mul vl]             \n"
        // Low nibble: shift left by 4 so the value lands pre-scaled by 16 in its natural sign position,
        // ready to feed smopa directly - no sign-extension or lane reordering needed.
        "   lsl     z24.b, z0.b, #4                          \n"
        "   lsl     z25.b, z1.b, #4                          \n"
        "   lsl     z26.b, z2.b, #4                          \n"
        "   lsl     z27.b, z3.b, #4                          \n"
        "   .inst 0xa0984a00 // smopa   za0.s, p2/m, p2/m, z16.b, z24.b\n"
        "   .inst 0xa0994a01 // smopa   za1.s, p2/m, p2/m, z16.b, z25.b\n"
        "   .inst 0xa09a4a02 // smopa   za2.s, p2/m, p2/m, z16.b, z26.b\n"
        "   .inst 0xa09b4a03 // smopa   za3.s, p2/m, p2/m, z16.b, z27.b\n"
        // High nibble: mask off the low nibble with 0xf0. This nibble packs the K-value 16 positions
        // ahead, already pre-scaled by 16 in its byte position, so no shift/zip is required here.
        "   and     z0.d, z0.d, z28.d                        \n"
        "   and     z1.d, z1.d, z28.d                        \n"
        "   and     z2.d, z2.d, z28.d                        \n"
        "   and     z3.d, z3.d, z28.d                        \n"
        "   .inst 0xa0804a80 // smopa   za0.s, p2/m, p2/m, z20.b, z0.b\n"
        "   .inst 0xa0814a81 // smopa   za1.s, p2/m, p2/m, z20.b, z1.b\n"
        "   .inst 0xa0824a82 // smopa   za2.s, p2/m, p2/m, z20.b, z2.b\n"
        "   .inst 0xa0834a83 // smopa   za3.s, p2/m, p2/m, z20.b, z3.b\n"
        "   ld1w    { z17.s }, p0/z, [x10, #0x1, mul vl]     \n"
        "   ld1w    { z21.s }, p0/z, [x10, #0x5, mul vl]     \n"
        "   addvl   x15, x11, #4                             \n"
        "   whilelt p3.h, x15, x28                           \n"
        "   addvl   x15, x11, #5                             \n"
        "   whilelt p4.h, x15, x28                           \n"
        "   addvl   x15, x11, #6                             \n"
        "   whilelt p6.h, x15, x28                           \n"
        "   addvl   x15, x11, #7                             \n"
        "   whilelt p7.h, x15, x28                           \n"
        "   ld1h    z4.h, p3/z, [x11, #4, mul vl]             \n"
        "   ld1h    z5.h, p4/z, [x11, #5, mul vl]             \n"
        "   ld1h    z6.h, p6/z, [x11, #6, mul vl]             \n"
        "   ld1h    z7.h, p7/z, [x11, #7, mul vl]             \n"
        "   addvl   x11, x11, #8                              \n"
        "   lsl     z24.b, z4.b, #4                          \n"
        "   lsl     z25.b, z5.b, #4                          \n"
        "   lsl     z26.b, z6.b, #4                          \n"
        "   lsl     z27.b, z7.b, #4                          \n"
        "   .inst 0xa0984a20 // smopa   za0.s, p2/m, p2/m, z17.b, z24.b\n"
        "   .inst 0xa0994a21 // smopa   za1.s, p2/m, p2/m, z17.b, z25.b\n"
        "   .inst 0xa09a4a22 // smopa   za2.s, p2/m, p2/m, z17.b, z26.b\n"
        "   .inst 0xa09b4a23 // smopa   za3.s, p2/m, p2/m, z17.b, z27.b\n"
        "   and     z4.d, z4.d, z28.d                        \n"
        "   and     z5.d, z5.d, z28.d                        \n"
        "   and     z6.d, z6.d, z28.d                        \n"
        "   and     z7.d, z7.d, z28.d                        \n"
        "   .inst 0xa0844aa0 // smopa   za0.s, p2/m, p2/m, z21.b, z4.b\n"
        "   .inst 0xa0854aa1 // smopa   za1.s, p2/m, p2/m, z21.b, z5.b\n"
        "   .inst 0xa0864aa2 // smopa   za2.s, p2/m, p2/m, z21.b, z6.b\n"
        "   .inst 0xa0874aa3 // smopa   za3.s, p2/m, p2/m, z21.b, z7.b\n"
        "   ld1w    { z18.s }, p0/z, [x10, #0x2, mul vl]     \n"
        "   ld1w    { z22.s }, p0/z, [x10, #0x6, mul vl]     \n"
        "   mov     x15, x11                                 \n"
        "   whilelt p3.h, x15, x28                           \n"
        "   addvl   x15, x11, #1                             \n"
        "   whilelt p4.h, x15, x28                           \n"
        "   addvl   x15, x11, #2                             \n"
        "   whilelt p6.h, x15, x28                           \n"
        "   addvl   x15, x11, #3                             \n"
        "   whilelt p7.h, x15, x28                           \n"
        "   ld1h    z8.h,  p3/z, [x11]                        \n"
        "   ld1h    z9.h,  p4/z, [x11, #1, mul vl]            \n"
        "   ld1h    z10.h, p6/z, [x11, #2, mul vl]            \n"
        "   ld1h    z11.h, p7/z, [x11, #3, mul vl]            \n"
        "   lsl     z24.b, z8.b, #4                          \n"
        "   lsl     z25.b, z9.b, #4                          \n"
        "   lsl     z26.b, z10.b, #4                         \n"
        "   lsl     z27.b, z11.b, #4                         \n"
        "   .inst 0xa0984a40 // smopa   za0.s, p2/m, p2/m, z18.b, z24.b\n"
        "   .inst 0xa0994a41 // smopa   za1.s, p2/m, p2/m, z18.b, z25.b\n"
        "   .inst 0xa09a4a42 // smopa   za2.s, p2/m, p2/m, z18.b, z26.b\n"
        "   .inst 0xa09b4a43 // smopa   za3.s, p2/m, p2/m, z18.b, z27.b\n"
        "   and     z8.d,  z8.d,  z28.d                      \n"
        "   and     z9.d,  z9.d,  z28.d                      \n"
        "   and     z10.d, z10.d, z28.d                      \n"
        "   and     z11.d, z11.d, z28.d                      \n"
        "   .inst 0xa0884ac0 // smopa   za0.s, p2/m, p2/m, z22.b, z8.b\n"
        "   .inst 0xa0894ac1 // smopa   za1.s, p2/m, p2/m, z22.b, z9.b\n"
        "   .inst 0xa08a4ac2 // smopa   za2.s, p2/m, p2/m, z22.b, z10.b\n"
        "   .inst 0xa08b4ac3 // smopa   za3.s, p2/m, p2/m, z22.b, z11.b\n"
        "   ld1w    { z19.s }, p0/z, [x10, #0x3, mul vl]     \n"
        "   ld1w    { z23.s }, p0/z, [x10, #0x7, mul vl]     \n"
        "   addvl   x15, x11, #4                             \n"
        "   whilelt p3.h, x15, x28                           \n"
        "   addvl   x15, x11, #5                             \n"
        "   whilelt p4.h, x15, x28                           \n"
        "   addvl   x15, x11, #6                             \n"
        "   whilelt p6.h, x15, x28                           \n"
        "   addvl   x15, x11, #7                             \n"
        "   whilelt p7.h, x15, x28                           \n"
        "   ld1h    z12.h, p3/z, [x11, #4, mul vl]            \n"
        "   ld1h    z13.h, p4/z, [x11, #5, mul vl]            \n"
        "   ld1h    z14.h, p6/z, [x11, #6, mul vl]            \n"
        "   ld1h    z15.h, p7/z, [x11, #7, mul vl]            \n"
        "   lsl     z24.b, z12.b, #4                         \n"
        "   lsl     z25.b, z13.b, #4                         \n"
        "   lsl     z26.b, z14.b, #4                         \n"
        "   lsl     z27.b, z15.b, #4                         \n"
        "   .inst 0xa0984a60 // smopa   za0.s, p2/m, p2/m, z19.b, z24.b\n"
        "   .inst 0xa0994a61 // smopa   za1.s, p2/m, p2/m, z19.b, z25.b\n"
        "   .inst 0xa09a4a62 // smopa   za2.s, p2/m, p2/m, z19.b, z26.b\n"
        "   .inst 0xa09b4a63 // smopa   za3.s, p2/m, p2/m, z19.b, z27.b\n"
        "   and     z12.d, z12.d, z28.d                      \n"
        "   and     z13.d, z13.d, z28.d                      \n"
        "   and     z14.d, z14.d, z28.d                      \n"
        "   and     z15.d, z15.d, z28.d                      \n"
        "   .inst 0xa08c4ae0 // smopa   za0.s, p2/m, p2/m, z23.b, z12.b\n"
        "   .inst 0xa08d4ae1 // smopa   za1.s, p2/m, p2/m, z23.b, z13.b\n"
        "   .inst 0xa08e4ae2 // smopa   za2.s, p2/m, p2/m, z23.b, z14.b\n"
        "   .inst 0xa08f4ae3 // smopa   za3.s, p2/m, p2/m, z23.b, z15.b\n"
        "   addvl   x11, x11, #8                              \n"
        "   addvl   x10, x10, #8                              \n"
        "   cmp     x10, x14                                 \n"
        "   b.lt    3b                                       \n"
        "   whilelt p4.s, x21, x13                            \n"
        "   incw  x21                                         \n"
        "   whilelt p5.s, x21, x13                            \n"
        "   incw  x21                                         \n"
        "   whilelt p6.s, x21, x13                            \n"
        "   incw  x21                                         \n"
        "   whilelt p7.s, x21, x13                            \n"
        "   decw  x21                                         \n"
        "   decw  x21                                         \n"
        "   decw  x21                                         \n"
        "   ld1w z0.s, p4/Z, [x11]                             \n"
        "   ld1w z1.s, p5/Z, [x11, #1, MUL VL]                 \n"
        "   ld1w z2.s, p6/Z, [x11, #2, MUL VL]                 \n"
        "   ld1w z3.s, p7/Z, [x11, #3, MUL VL]                 \n"
        "   addvl x11, x11, #4                                 \n"
        "   ld1w z4.s, p4/Z, [x11]                             \n"
        "   ld1w z5.s, p5/Z, [x11, #1, MUL VL]                 \n"
        "   ld1w z6.s, p6/Z, [x11, #2, MUL VL]                 \n"
        "   ld1w z7.s, p7/Z, [x11, #3, MUL VL]                 \n"
        "   addvl x11, x11, #4                                 \n"
        "   ld1w z8.s, p4/Z, [x11]                             \n"
        "   ld1w z9.s, p5/Z, [x11, #1, MUL VL]                 \n"
        "   ld1w z10.s, p6/Z, [x11, #2, MUL VL]                \n"
        "   ld1w z11.s, p7/Z, [x11, #3, MUL VL]                \n"
        "   addvl x11, x11, #4                                 \n"
        "   mov     x14, #0                \n"
        "   addvl   x15, x10, #1           \n"
        "4:                                \n"
        "   ld1rw   {z16.s},  p2/z, [x10]  \n"
        "   ld1rw   {z17.s}, p2/z, [x15]   \n"
        "   add     x10, x10, #4           \n"
        "   add     x15, x15, #4           \n"
        "   fmul    z20.s, z17.s, z4.s    \n"
        "   fmul    z21.s, z17.s, z5.s    \n"
        "   fmul    z22.s, z17.s, z6.s    \n"
        "   fmul    z23.s, z17.s, z7.s    \n"
        "   .inst 0xc002480c // mova z12.b, p2/M, za0h.b[w14, 0]\n"
        "   .inst 0xc002482d // mova z13.b, p2/M, za0h.b[w14, 1]\n"
        "   .inst 0xc002484e // mova z14.b, p2/M, za0h.b[w14, 2]\n"
        "   .inst 0xc002486f // mova z15.b, p2/M, za0h.b[w14, 3]\n"
        "   mla     z12.s, p2/m, z16.s, z0.s  \n"
        "   mla     z13.s, p2/m, z16.s, z1.s  \n"
        "   mla     z14.s, p2/m, z16.s, z2.s  \n"
        "   mla     z15.s, p2/m, z16.s, z3.s  \n"
        "   scvtf z12.s, p2/M, z12.s                            \n"
        "   scvtf z13.s, p2/M, z13.s                            \n"
        "   scvtf z14.s, p2/M, z14.s                            \n"
        "   scvtf z15.s, p2/M, z15.s                            \n"
        "   fmul    z24.s, z12.s, z20.s   \n"
        "   fmul    z25.s, z13.s, z21.s   \n"
        "   fmul    z26.s, z14.s, z22.s   \n"
        "   fmul    z27.s, z15.s, z23.s   \n"
        "   fadd    z24.s, p2/m, z24.s, z8.s  \n"
        "   fadd    z25.s, p2/m, z25.s, z9.s  \n"
        "   fadd    z26.s, p2/m, z26.s, z10.s \n"
        "   fadd    z27.s, p2/m, z27.s, z11.s \n"
        "   fmin z24.s, p2/M, z24.s, z31.s                      \n"
        "   fmin z25.s, p2/M, z25.s, z31.s                      \n"
        "   fmin z26.s, p2/M, z26.s, z31.s                      \n"
        "   fmin z27.s, p2/M, z27.s, z31.s                      \n"
        "   fmax z24.s, p2/M, z24.s, z30.s                      \n"
        "   fmax z25.s, p2/M, z25.s, z30.s                      \n"
        "   fmax z26.s, p2/M, z26.s, z30.s                      \n"
        "   fmax z27.s, p2/M, z27.s, z30.s                      \n"
        "   st1w z24.s, p4, [x17]                               \n"
        "   st1w z25.s, p5, [x17, #1, MUL VL]                   \n"
        "   st1w z26.s, p6, [x17, #2, MUL VL]                   \n"
        "   st1w z27.s, p7, [x17, #3, MUL VL]                   \n"
        "   add     x17, x17, %[dst_stride_row]              \n"
        "   add     x14, x14, #4                             \n"
        "   cmp     x14, x16                                 \n"
        "   b.lt    4b                                       \n"
        "   add   x8, x8, %[rhs_stride]                      \n"
        "   addvl x9, x9, #4                                   \n"
        "   sub x13, x13, %[nr]                              \n"
        "   add     x28, x8, %[rhs_row_bytes]                \n"
        "   whilelt p1.h, xzr, x13                              \n"
        "   b.mi  2b                                         \n"
        "   add   x20, x20, %[lhs_stride]                    \n"
        "   add   x19, x19, %[dst_inc]                       \n"
        "   sub   x12, x12, %[mr]                            \n"
        "   whilelt p0.s, xzr, x12                           \n"
        "   b.mi 1b                                          \n"
        "5:                                                  \n"
        "   .inst 0xd503467f // smstop\n"
        :
        : [m] "r"(m), [n] "r"(n), [lhs_stride] "r"(lhs_stride), [rhs_stride] "r"(rhs_stride),
          [dst_stride_row] "r"(dst_stride_row), [m_blk] "r"(m_blk), [nr] "r"(nr), [mr] "r"(mr),
          [lhs] "r"(lhs_packed), [rhs] "r"(rhs_packed), [dst_inc] "r"(dst_inc), [scalar_bounds] "r"(scalar_bounds),
          [rhs_row_bytes] "r"(rhs_row_bytes), [dst] "r"(dst)
        : "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x19", "x20", "x21", "x28", "p0",
          "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8",
          "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23",
          "z24", "z25", "z26", "z27", "z28", "z30", "z31",
#ifdef __ARM_STATE_ZA
          "za",
#endif
#ifdef __ARM_STATE_ZT0
          "zt0",
#endif
          "cc", "memory");
}

#endif  // Architectural feature check
