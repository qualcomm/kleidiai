//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
__asm__(".include \"helper_macros_imatmul_clamp_f32_f32p_f32p.S\"");

static const size_t kai_mr = 2;
static const size_t kai_nr = 2;
static const size_t kai_kr = 1;

size_t kai_get_m_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_lhs_packed_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(
    size_t m_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(m_idx % kai_get_m_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa() == 0);
    const size_t indirect_k = k_chunk_count * kai_roundup(k_chunk_length, kai_kr);
    return m_idx * indirect_k * sizeof(float);
}

static size_t kai_get_rhs_packed_stride_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(
    size_t k_chunk_count, size_t k_chunk_length) {
    const size_t indirect_k = k_chunk_count * kai_roundup(k_chunk_length, kai_kr);
    return kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa() *
        (sizeof(float) + indirect_k * sizeof(float));
}

size_t kai_get_rhs_packed_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(n_idx % kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa() == 0);
    const size_t block_idx = n_idx / kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa();
    return block_idx *
        kai_get_rhs_packed_stride_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(
               k_chunk_count, k_chunk_length);
}

size_t kai_get_dst_offset_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(
    size_t m_idx, size_t n_idx, size_t dst_row_stride) {
    KAI_ASSUME(m_idx % kai_get_m_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa() == 0);

    return m_idx * dst_row_stride + n_idx * sizeof(float);
}

size_t kai_get_dst_size_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa(
    size_t m, size_t n, size_t k_chunk_count, size_t k_chunk_length, const void* lhs_packed, const void* rhs_packed,
    void* dst, size_t dst_row_stride, float clamp_min, float clamp_max) {
    typedef struct {
        const void* A;
        const void* B;
        void* C;
        uint64_t ldcb;
        uint64_t M;
        uint64_t N;
        uint64_t K;
        float min;
        float max;
        void* accumulator_buffer;
        uint64_t flags;
    } KernelArgs;

    KernelArgs args;

    args.A = lhs_packed;
    args.B = rhs_packed;

    const size_t indirect_k = k_chunk_count * kai_roundup(k_chunk_length, kai_kr);

    args.C = dst;
    args.ldcb = dst_row_stride;
    args.M = m;
    args.N = n;
    args.K = indirect_k;
    args.min = clamp_min;
    args.max = clamp_max;

    args.accumulator_buffer = NULL;
    args.flags = 0;

    __asm__ __volatile__(
        "SMSTART\n"
        "ldr w14, [%x[args], %[offsetof_M]]\n"
        "mov x13, #0x0\n"
        "mov x11, #0x0\n"
        "ptrue p0.b\n"
        "ptrue p5.b\n"
        "ldr w10, [%x[args], %[offsetof_N]]\n"
        "ldr x9, [%x[args], %[offsetof_A]]\n"
        "1:"  // M loop
        "ldr x28, [%x[args], %[offsetof_B]]\n"
        "2:"  // N loop
        "whilelt p4.s, x11, x10\n"
        "fmov z13.s, #1.0\n"
        "zero { za }\n"
        "mov x27, x9\n"
		"load_data_pred_index_counter_2 z14.s, z15.s, p4, p4.s, x11, x10, x28, x15\n" // Load bias
        "addvl x28, x28, #2\n"
        "fmopa za0.s, p0/M, p0/M, z13.s, z14.s\n"
        "fmopa za1.s, p0/M, p0/M, z13.s, z15.s\n"
        "fmopa za2.s, p0/M, p0/M, z13.s, z14.s\n"
        "fmopa za3.s, p0/M, p0/M, z13.s, z15.s\n"
        "ldr x20, [%x[args], %[offsetof_K]]\n"
        "lsr x21, x20, #0x2\n"
        "and x20, x20, #0x3\n"
        "cbz x21, 6f\n"
        "subs x21, x21, #0x1\n"
		"load_data_2 z18.s, z26.s, p5, x27\n"
		"load_data_2 z20.s, z21.s, p5, x28\n"
		"load_data_2 z4.s, z12.s, p5, x27\n"
		"load_data_2 z10.s, z11.s, p5, x28\n"
		"load_data_2 z19.s, z27.s, p5, x27\n"
		"load_data_2 z24.s, z25.s, p5, x28\n"
		"load_data_2 z14.s, z15.s, p5, x27\n"
		"load_data_2 z22.s, z30.s, p5, x28\n"
        "ble 5f\n"
        "4:"  // K loop
        "fmopa za0.s, p0/M, p0/M, z18.s, z20.s\n"
        "subs x21, x21, #0x1\n"
        "fmopa za1.s, p0/M, p0/M, z18.s, z21.s\n"
        "fmopa za2.s, p0/M, p0/M, z26.s, z20.s\n"
        "fmopa za3.s, p0/M, p0/M, z26.s, z21.s\n"
		"load_data_2 z18.s, z26.s, p5, x27\n"
        "fmopa za0.s, p0/M, p0/M, z4.s, z10.s\n"
		"load_data_2 z20.s, z21.s, p5, x28\n"
        "fmopa za1.s, p0/M, p0/M, z4.s, z11.s\n"
        "fmopa za2.s, p0/M, p0/M, z12.s, z10.s\n"
        "fmopa za3.s, p0/M, p0/M, z12.s, z11.s\n"
		"load_data_2 z4.s, z12.s, p5, x27\n"
        "fmopa za0.s, p0/M, p0/M, z19.s, z24.s\n"
		"load_data_2 z10.s, z11.s, p5, x28\n"
        "fmopa za1.s, p0/M, p0/M, z19.s, z25.s\n"
        "fmopa za2.s, p0/M, p0/M, z27.s, z24.s\n"
        "fmopa za3.s, p0/M, p0/M, z27.s, z25.s\n"
		"load_data_2 z19.s, z27.s, p5, x27\n"
		"load_data_2 z24.s, z25.s, p5, x28\n"
        "fmopa za0.s, p0/M, p0/M, z14.s, z22.s\n"
        "fmopa za1.s, p0/M, p0/M, z14.s, z30.s\n"
        "fmopa za2.s, p0/M, p0/M, z15.s, z22.s\n"
        "fmopa za3.s, p0/M, p0/M, z15.s, z30.s\n"
		"load_data_2 z14.s, z15.s, p5, x27\n"
		"load_data_2 z22.s, z30.s, p5, x28\n"
        "bgt 4b\n"
        "5:"  // K loop tail
        "fmopa za0.s, p0/M, p0/M, z18.s, z20.s\n"
        "fmopa za1.s, p0/M, p0/M, z18.s, z21.s\n"
        "fmopa za2.s, p0/M, p0/M, z26.s, z20.s\n"
        "fmopa za3.s, p0/M, p0/M, z26.s, z21.s\n"
        "fmopa za0.s, p0/M, p0/M, z4.s, z10.s\n"
        "fmopa za1.s, p0/M, p0/M, z4.s, z11.s\n"
        "fmopa za2.s, p0/M, p0/M, z12.s, z10.s\n"
        "fmopa za3.s, p0/M, p0/M, z12.s, z11.s\n"
        "fmopa za0.s, p0/M, p0/M, z19.s, z24.s\n"
        "fmopa za1.s, p0/M, p0/M, z19.s, z25.s\n"
        "fmopa za2.s, p0/M, p0/M, z27.s, z24.s\n"
        "fmopa za3.s, p0/M, p0/M, z27.s, z25.s\n"
        "fmopa za0.s, p0/M, p0/M, z14.s, z22.s\n"
        "fmopa za1.s, p0/M, p0/M, z14.s, z30.s\n"
        "fmopa za2.s, p0/M, p0/M, z15.s, z22.s\n"
        "fmopa za3.s, p0/M, p0/M, z15.s, z30.s\n"
        "6:"  // K oddments
        "cbz x20, 8f\n"
        "7:"  // K oddments: Loop
		"load_data_2 z28.s, z29.s, p5, x27\n"
        "subs x20, x20, #0x1\n"
		"load_data_2 z7.s, z15.s, p5, x28\n"
        "fmopa za0.s, p0/M, p0/M, z28.s, z7.s\n"
        "fmopa za1.s, p0/M, p0/M, z28.s, z15.s\n"
        "fmopa za2.s, p0/M, p0/M, z29.s, z7.s\n"
        "fmopa za3.s, p0/M, p0/M, z29.s, z15.s\n"
        "bgt 7b\n"
        "8:"  // K oddments: End
        "ldr x26, [%x[args], %[offsetof_C]]\n"
        "sub x25, x14, x13\n"
        "cntw x24\n"
        "ld1rw { z19.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
        "ldr x23, [%x[args], %[offsetof_ldcb]]\n"
        "cmp x25, x24\n"
        "ld1rw { z26.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
        "mov x12, #0x0\n"
        "csel x22, x25, x24, LT\n"
        "add x26, x26, x11, LSL #2\n"  // C += n
        "lsr x21, x22, #0x2\n"
        "madd x26, x13, x23, x26\n"  // C += m * ldc
        "and x20, x22, #0x3\n"
        "cbz x21, 11f\n"
        "10:"  // Store to output array: Accumulator row 0 loop
		"move_tile_vector za0h.s, z4.s, z5.s, z6.s, z7.s, p5, w12\n"
		"move_tile_vector za1h.s, z12.s, z13.s, z14.s, z15.s, p5, w12\n"
		"clamp_float z4.s, z26.s, z19.s, p5\n"
		"clamp_float z5.s, z26.s, z19.s, p5\n"
		"clamp_float z6.s, z26.s, z19.s, p5\n"
		"clamp_float z7.s, z26.s, z19.s, p5\n"
		"clamp_float z12.s, z26.s, z19.s, p5\n"
		"clamp_float z13.s, z26.s, z19.s, p5\n"
		"clamp_float z14.s, z26.s, z19.s, p5\n"
		"clamp_float z15.s, z26.s, z19.s, p5\n"
        "add x12, x12, #0x4\n"
		"store_data_pred_index_counter_2 z4.s, z12.s, p4, p4.s, x11, x10, x26, x15\n"
        "add x26, x26, x23\n"
		"store_data_pred_index_counter_2 z5.s, z13.s, p4, p4.s, x11, x10, x26, x15\n"
        "add x26, x26, x23\n"
		"store_data_pred_index_counter_2 z6.s, z14.s, p4, p4.s, x11, x10, x26, x15\n"
        "add x26, x26, x23\n"
		"store_data_pred_index_counter_2 z7.s, z15.s, p4, p4.s, x11, x10, x26, x15\n"
        "add x26, x26, x23\n"
        "cmp x12, x21, LSL #2\n"
        "blt 10b\n"
        "11:"  // Store to output array: Accumulator row 0 oddments
        "cbz x20, 12f\n"
		"move_tile_vector za0h.s, z0.s, z1.s, z2.s, z3.s, p5, w12\n"
		"move_tile_vector za1h.s, z8.s, z9.s, z10.s, z11.s, p5, w12\n"
		"clamp_float z0.s, z26.s, z19.s, p5\n"
		"clamp_float z1.s, z26.s, z19.s, p5\n"
		"clamp_float z2.s, z26.s, z19.s, p5\n"
		"clamp_float z3.s, z26.s, z19.s, p5\n"
		"clamp_float z8.s, z26.s, z19.s, p5\n"
		"clamp_float z9.s, z26.s, z19.s, p5\n"
		"clamp_float z10.s, z26.s, z19.s, p5\n"
		"clamp_float z11.s, z26.s, z19.s, p5\n"
		"store_data_pred_index_counter_2 z0.s, z8.s, p4, p4.s, x11, x10, x26, x15\n"
        "add x26, x26, x23\n"
        "subs x20, x20, #0x1\n"
        "beq 12f\n"
		"store_data_pred_index_counter_2 z1.s, z9.s, p4, p4.s, x11, x10, x26, x15\n"
        "add x26, x26, x23\n"
        "subs x20, x20, #0x1\n"
        "beq 12f\n"
		"store_data_pred_index_counter_2 z2.s, z10.s, p4, p4.s, x11, x10, x26, x15\n"
        "add x26, x26, x23\n"
        "12:"  // Store to output array: Accumulator row 0 oddments: End
        "subs x25, x25, x22\n"
        "beq 16f\n"
        "cmp x25, x24\n"
        "mov x12, #0x0\n"
        "csel x20, x25, x24, LT\n"
        "lsr x21, x20, #0x2\n"
        "and x20, x20, #0x3\n"
        "cbz x21, 14f\n"
        "13:"  // Store to output array: Accumulator row 1 loop
		"move_tile_vector za2h.s, z20.s, z21.s, z22.s, z23.s, p5, w12\n"
		"move_tile_vector za3h.s, z28.s, z29.s, z30.s, z31.s, p5, w12\n"
		"clamp_float z20.s, z26.s, z19.s, p5\n"
		"clamp_float z21.s, z26.s, z19.s, p5\n"
		"clamp_float z22.s, z26.s, z19.s, p5\n"
		"clamp_float z23.s, z26.s, z19.s, p5\n"
		"clamp_float z28.s, z26.s, z19.s, p5\n"
		"clamp_float z29.s, z26.s, z19.s, p5\n"
		"clamp_float z30.s, z26.s, z19.s, p5\n"
		"clamp_float z31.s, z26.s, z19.s, p5\n"
        "add x12, x12, #0x4\n"
		"store_data_pred_index_counter_2 z20.s, z28.s, p4, p4.s, x11, x10, x26, x15\n"
        "add x26, x26, x23\n"
		"store_data_pred_index_counter_2 z21.s, z29.s, p4, p4.s, x11, x10, x26, x15\n"
        "add x26, x26, x23\n"
		"store_data_pred_index_counter_2 z22.s, z30.s, p4, p4.s, x11, x10, x26, x15\n"
        "add x26, x26, x23\n"
		"store_data_pred_index_counter_2 z23.s, z31.s, p4, p4.s, x11, x10, x26, x15\n"
        "add x26, x26, x23\n"
        "cmp x12, x21, LSL #2\n"
        "blt 13b\n"
        "14:"  // Store to output array: Accumulator row 1 oddments
        "cbz x20, 15f\n"
		"move_tile_vector za2h.s, z4.s, z5.s, z6.s, z7.s, p5, w12\n"
		"move_tile_vector za3h.s, z12.s, z13.s, z14.s, z15.s, p5, w12\n"
		"clamp_float z4.s, z26.s, z19.s, p5\n"
		"clamp_float z5.s, z26.s, z19.s, p5\n"
		"clamp_float z6.s, z26.s, z19.s, p5\n"
		"clamp_float z7.s, z26.s, z19.s, p5\n"
		"clamp_float z12.s, z26.s, z19.s, p5\n"
		"clamp_float z13.s, z26.s, z19.s, p5\n"
		"clamp_float z14.s, z26.s, z19.s, p5\n"
		"clamp_float z15.s, z26.s, z19.s, p5\n"
		"store_data_pred_index_counter_2 z4.s, z12.s, p4, p4.s, x11, x10, x26, x15\n"
        "add x26, x26, x23\n"
		"subs x20, x20, #0x1\n"
        "beq 15f\n"
		"store_data_pred_index_counter_2 z5.s, z13.s, p4, p4.s, x11, x10, x26, x15\n"
        "add x26, x26, x23\n"
		"subs x20, x20, #0x1\n"
        "beq 15f\n"
		"store_data_pred_index_counter_2 z6.s, z14.s, p4, p4.s, x11, x10, x26, x15\n"
        "15:"  // Store to output array: Accumulator row 1 oddments: End
        "16:"  // Store to output array: End
        "incw x11, ALL, MUL #2\n"
        "cmp x11, x10\n"
        "blt 2b\n"
        "incw x13, ALL, MUL #2\n"
        "mov x11, #0x0\n"
        "cmp x13, x14\n"
        "mov x9, x27\n"
        "blt 1b\n"
        "SMSTOP\n"
        :
        : [args] "r"(&args), [offsetof_A] "I"(offsetof(KernelArgs, A)), [offsetof_B] "I"(offsetof(KernelArgs, B)),
          [offsetof_C] "I"(offsetof(KernelArgs, C)), [offsetof_K] "I"(offsetof(KernelArgs, K)),
          [offsetof_KernelArgs_max] "I"(offsetof(KernelArgs, max)),
          [offsetof_KernelArgs_min] "I"(offsetof(KernelArgs, min)), [offsetof_M] "I"(offsetof(KernelArgs, M)),
          [offsetof_N] "I"(offsetof(KernelArgs, N)), [offsetof_ldcb] "I"(offsetof(KernelArgs, ldcb))
        : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7",
          "p4", "p9", "x10", "x11", "x12", "x13", "x14", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28",
          "x9", "z0", "z1", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z2", "z20", "z21",
          "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z3", "z30", "z31", "z4", "z5", "z6", "z7", "z8",
          "z9", "x15");
}

#endif  // Architectural features check.
