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

#include "kai_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"
__asm__(".include \"helper_macros_imatmul_clamp_f16_f16p_f16p.S\"");

static const size_t kai_mr = 2;
static const size_t kai_nr = 2;
static const size_t kai_kr = 2;

size_t kai_get_m_step_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_lhs_packed_offset_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa(
    size_t m_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(m_idx % kai_get_m_step_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa() == 0);
    const size_t indirect_k = k_chunk_count * kai_roundup(k_chunk_length, kai_kr);
    return m_idx * indirect_k * sizeof(uint16_t);
}

static size_t kai_get_rhs_packed_stride_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa(
    size_t k_chunk_count, size_t k_chunk_length) {
    const size_t indirect_k = k_chunk_count * kai_roundup(k_chunk_length, kai_kr);
    return kai_get_n_step_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa() *
        (sizeof(uint16_t) + indirect_k * sizeof(uint16_t));
}

size_t kai_get_rhs_packed_offset_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa(
    size_t n_idx, size_t k_chunk_count, size_t k_chunk_length) {
    KAI_ASSUME(n_idx % kai_get_n_step_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa() == 0);
    const size_t block_idx = n_idx / kai_get_n_step_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa();
    return block_idx *
        kai_get_rhs_packed_stride_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa(
               k_chunk_count, k_chunk_length);
}

size_t kai_get_dst_offset_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa(
    size_t m_idx, size_t n_idx, size_t dst_row_stride) {
    KAI_ASSUME(m_idx % kai_get_m_step_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa() == 0);

    return m_idx * dst_row_stride + n_idx * sizeof(uint16_t);
}

size_t kai_get_dst_size_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa(size_t m, size_t n) {
    return m * n * sizeof(uint16_t);
}

void kai_run_imatmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme1_mopa(
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
        float16_t min;
        float16_t max;
        void* accumulator_buffer;
        uint64_t flags;
    } KernelArgs;

    KernelArgs args;

    args.A = lhs_packed;
    args.B = rhs_packed;

    size_t indirect_k = k_chunk_count * kai_roundup(k_chunk_length, kai_kr);

    args.C = dst;
    args.ldcb = dst_row_stride;
    args.M = m;
    args.N = n;
    args.K = indirect_k;
    args.min = (float16_t)clamp_min;
    args.max = (float16_t)clamp_max;

    args.accumulator_buffer = NULL;
    args.flags = 0;

    __asm__ __volatile__(
    "SMSTART \n"
        "ldr w13, [%x[args], %[offsetof_M]]\n"
        "mov x11, #0x0\n"
        "mov x10, #0x0\n"
        "ptrue p1.b\n"
        " ptrue p2.b\n"
        "ldr w9, [%x[args], %[offsetof_N]]\n"
        "ldr x28, [%x[args], %[offsetof_A]]\n"
        "1:"  // M loop
        "ldr x27, [%x[args], %[offsetof_B]]\n"
        "2:"  // N loop
        "fmov z24.h, #0.0\n"
        "ld1h { z5.h }, p1/Z, [x27]\n"
        "fmov z27.h, #1.0\n"
        "mov x26, x28\n"
        "zero { za }\n"
        "inch x27, ALL, MUL #2\n"
        "zip1 z30.h, z5.h, z24.h\n"
        "zip2 z20.h, z5.h, z24.h\n"
        "fmopa za0.s, p1/M, p1/M, z27.h, z30.h\n"
        "fmopa za1.s, p1/M, p1/M, z27.h, z20.h\n"
        "fmopa za2.s, p1/M, p1/M, z27.h, z30.h\n"
        "fmopa za3.s, p1/M, p1/M, z27.h, z20.h\n"
        "ldr x20, [%x[args], %[offsetof_K]]\n"
        "add x20, x20, #0x1\n"
        "lsr x20, x20, #0x1\n"
        "lsr x21, x20, #0x2\n"
        "and x20, x20, #0x3\n"
        "cbz x21, 6f\n"
        "subs x21, x21, #0x1\n"	
        "ld1h_data_2  z18.h,z19.h , p2, x26 \n"
        "ld1h_data_2  z16.h,z17.h , p2, x27 \n"
        "ld1h_data_2  z2.h,z10.h , p2, x26 \n"
        "ld1h_data_2  z30.h,z31.h , p2, x27 \n"
        "ld1h_data_2  z28.h,z29.h , p2, x26 \n"
        "ld1h_data_2  z6.h,z14.h , p2, x27 \n"
        "ld1h_data_2  z5.h, z13.h , p2, x26 \n"
        "ld1h_data_2  z7.h, z15.h , p2, x27 \n"				  
        "ble 5f\n"
        "4:"  // K loop
        " fmopa za0.s, p1/M, p1/M, z18.h, z16.h\n"
        " fmopa za1.s, p1/M, p1/M, z18.h, z17.h\n"
        " fmopa za2.s, p1/M, p1/M, z19.h, z16.h\n"
        " fmopa za3.s, p1/M, p1/M, z19.h, z17.h\n"
        " ld1h_data_2  z18.h,z19.h , p2, x26 \n"
        " fmopa za0.s, p1/M, p1/M, z2.h, z30.h\n"
        " ld1h_data_2  z16.h,z17.h , p2, x27 \n"
        " fmopa za1.s, p1/M, p1/M, z2.h, z31.h\n"
        " fmopa za2.s, p1/M, p1/M, z10.h, z30.h\n"
        " fmopa za3.s, p1/M, p1/M, z10.h, z31.h\n"
        " ld1h_data_2  z2.h, z10.h , p2, x26 \n"
        " fmopa za0.s, p1/M, p1/M, z28.h, z6.h\n"
        " ld1h_data_2  z30.h,z31.h , p2, x27 \n"
        " fmopa za1.s, p1/M, p1/M, z28.h, z14.h\n"
        " fmopa za2.s, p1/M, p1/M, z29.h, z6.h\n"
        " fmopa za3.s, p1/M, p1/M, z29.h, z14.h\n"
        " ld1h_data_2  z28.h,z29.h , p2, x26 \n"
        " ld1h_data_2  z6.h, z14.h , p2, x27 \n"
        " fmopa za0.s, p1/M, p1/M, z5.h, z7.h\n"
        " fmopa za1.s, p1/M, p1/M, z5.h, z15.h\n"
        " fmopa za2.s, p1/M, p1/M, z13.h, z7.h\n"
        " fmopa za3.s, p1/M, p1/M, z13.h, z15.h\n"
        " ld1h_data_2  z5.h, z13.h , p2, x26 \n"
        " ld1h_data_2  z7.h, z15.h , p2, x27 \n"
        "subs x21, x21, #0x1\n"							  
        "bgt 4b\n"
        "5:"  // K loop tail
        " fmopa za0.s, p1/M, p1/M, z18.h, z16.h\n"
        " fmopa za1.s, p1/M, p1/M, z18.h, z17.h\n"
        " fmopa za2.s, p1/M, p1/M, z19.h, z16.h\n"
        " fmopa za3.s, p1/M, p1/M, z19.h, z17.h\n"
        " fmopa za0.s, p1/M, p1/M, z2.h, z30.h\n"
        " fmopa za1.s, p1/M, p1/M, z2.h, z31.h\n"
        " fmopa za2.s, p1/M, p1/M, z10.h, z30.h\n"
        " fmopa za3.s, p1/M, p1/M, z10.h, z31.h\n"
        " fmopa za0.s, p1/M, p1/M, z28.h, z6.h\n"
        " fmopa za1.s, p1/M, p1/M, z28.h, z14.h\n"
        " fmopa za2.s, p1/M, p1/M, z29.h, z6.h\n"
        " fmopa za3.s, p1/M, p1/M, z29.h, z14.h\n"
        " fmopa za0.s, p1/M, p1/M, z5.h, z7.h\n"
        " fmopa za1.s, p1/M, p1/M, z5.h, z15.h\n"
        " fmopa za2.s, p1/M, p1/M, z13.h, z7.h\n"
        " fmopa za3.s, p1/M, p1/M, z13.h, z15.h\n"
        "6:"  // K oddments
        "cbz x20, 8f\n"
        "7:"  // K oddments: Loop
        " ld1h_data_2  z5.h, z13.h , p2, x26\n"
		"subs x20, x20, #0x1\n"
        " ld1h_data_2  z14.h,z15.h , p2, x27\n"	  
        " fmopa za0.s, p1/M, p1/M, z5.h, z14.h\n"
        " fmopa za1.s, p1/M, p1/M, z5.h, z15.h\n"
        " fmopa za2.s, p1/M, p1/M, z13.h, z14.h\n"
        " fmopa za3.s, p1/M, p1/M, z13.h, z15.h\n"
        "bgt 7b\n"
        "8:"  // K oddments: End
        "ldr x25, [%x[args], %[offsetof_C]]\n"
        "sub x24, x13, x11\n"
        "cntw x23, ALL, MUL #2\n"
        "ld1rh { z17.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
        "ldr x22, [%x[args], %[offsetof_ldcb]]\n"
        "whilelt p0.h, x10, x9\n"
        "cmp x24, x23\n"
        "ld1rh { z16.h }, p1/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
        "mov x12, #0x0\n"
        "mov x21, #0x0\n"
        "add x25, x25, x10, LSL #1\n"  // C += n
        "mov x20, #0x2\n"
        "madd x25, x11, x22, x25\n"  // C += m * ldc
        "csel x24, x24, x23, LT\n"
        "10:"  // Store to output array: Accumulator loop
        "mova z14.b,p1/M, za0h.b[w12, 0]\n"
        "mova z15.b,p1/M, za0h.b[w12, 1]\n"
		"add x12, x12, #0x4\n"
        "cmp x12, x23, LSL #1\n"
        "add x21, x21, #0x1\n"
        "fcvt z0.h, p1/M,   z14.s\n"
        "fcvt z1.h, p1/M,   z15.s\n"
        "uzp1 z12.h ,z0.h,  z1.h\n"
        "csel x12, x12, x20, LT\n"
        "clamp_float z12.h, z17.h, z16.h, p1 \n"
        "st1h { z12.h }, p0, [x25]\n"
        "add x25, x25, x22\n"
        "cmp x21, x24\n"
        "blt 10b\n"
        "incw x10, ALL, MUL #2\n"
        "cmp x10, x9\n"
        "blt 2b\n"
        "incw x11, ALL, MUL #2\n"
        "mov x10, #0x0\n"
		"cmp x11, x13\n"
        "mov x28, x26\n"
        "blt 1b\n"
        "SMSTOP\n"
        :
        : [args] "r"(&args), [offsetof_A] "I"(offsetof(KernelArgs, A)), [offsetof_B] "I"(offsetof(KernelArgs, B)),
          [offsetof_C] "I"(offsetof(KernelArgs, C)), [offsetof_K] "I"(offsetof(KernelArgs, K)),
          [offsetof_KernelArgs_max] "I"(offsetof(KernelArgs, max)),
          [offsetof_KernelArgs_min] "I"(offsetof(KernelArgs, min)), [offsetof_M] "I"(offsetof(KernelArgs, M)),
          [offsetof_N] "I"(offsetof(KernelArgs, N)), [offsetof_ldcb] "I"(offsetof(KernelArgs, ldcb))
        : "cc", "memory", "p0", "p1", "p10", "p11", "p12", "p13", "p14", "p15", "p2", "p3", "p4", "p5", "p6", "p7",
          "p8", "p9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x9",
          "z0", "z1", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z2", "z20", "z21", "z22",
          "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z3", "z30", "z31", "z4", "z5", "z6", "z7", "z8", "z9");
}

#endif  // Architectural features check.
