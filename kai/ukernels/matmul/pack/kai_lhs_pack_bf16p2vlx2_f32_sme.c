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
#include "kai_lhs_pack_bf16p2vlx2_f32_sme.h"

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"


enum {
    MR = 2,
    KR = 2,
    MAX_M_STEP = MR * (KAI_SME_VEC_LENGTH_MAX_BYTES / sizeof(uint16_t)) / KR,
    SR = 1,
};

static size_t kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme(void) {
    return MR * kai_get_sme_vector_length_u16() / KR;
}

size_t kai_get_m_step_lhs_pack_bf16p2vlx2_f32_sme(size_t mr) {
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme());
    KAI_UNUSED(mr);
    return kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme();
}

size_t kai_get_lhs_offset_lhs_pack_bf16p2vlx2_f32_sme(size_t m_idx, size_t lhs_stride_row) {
    KAI_ASSUME(m_idx % kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme() == 0);

    return m_idx * lhs_stride_row;
}

size_t kai_get_lhs_packed_offset_lhs_pack_bf16p2vlx2_f32_sme(size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(m_idx % kai_get_m_step_lhs_pack_bf16p2vlx2_f32_sme(mr) == 0);
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == SR);

    KAI_UNUSED(mr);
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);

    return m_idx * kai_roundup(k, KR) * sizeof(uint16_t);
}

size_t kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme(size_t m, size_t k, size_t mr, size_t kr, size_t sr) {
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == SR);

    KAI_UNUSED(mr);
    KAI_UNUSED(kr);
    KAI_UNUSED(sr);
    return kai_roundup(m, kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme()) * kai_roundup(k, KR) * sizeof(uint16_t);
}

void kai_run_lhs_pack_bf16p2vlx2_f32_sme(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride_row,
    void* lhs_packed) {
    KAI_ASSUME(mr == kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme());
    KAI_ASSUME(kr == KR);
    KAI_ASSUME(sr == SR);
    KAI_ASSUME(m_idx_start == 0);
    KAI_ASSUME(lhs != NULL);
    KAI_ASSUME(lhs_packed != NULL);

    const size_t m_step = kai_get_mr_lhs_pack_bf16p2vlx2_f32_sme();
    const size_t width = k;

    KAI_ASSERT(m_step <= MAX_M_STEP);
    const uint8_t* in[MAX_M_STEP];

    uint8_t* out_base = lhs_packed;
    const uint8_t* lhs_ptr = lhs;

    kai_commit_za();

    for (size_t i_m = 0; i_m < m; i_m += m_step) {
        const size_t height = KAI_MIN(m - i_m, m_step);
        void* out = out_base;
        out_base += m_step * kai_roundup(k, KR) * sizeof(uint16_t);

        for (size_t y = 0; y < height; y++) {
            in[y] = lhs_ptr + (i_m + y) * lhs_stride_row;
        }

        __asm__ __volatile__(
            ".inst 0xd503477f  // SMSTART ZA\n"
            "sub x10, %x[width], #0x1\n"
            "mov x9, #0x0\n"
            "cntw x22, ALL, MUL #2\n"
            "cntw x28\n"
            "cntw x21, ALL, MUL #2\n"
            "sub x20, x22, #0x1\n"
            "ptrue p1.b\n"
            "whilelt p12.s, XZR, %x[height]\n"
            "whilelt p11.s, x28, %x[height]\n"
            "add x10, x10, x21\n"
            "ands x27, %x[width], x20\n"
            "udiv x10, x10, x21\n"
            "csel x27, x27, x22, NE\n"
            "and x26, x10, #0x1\n"
            "sub x10, x10, #0x1\n"
            "add x27, x27, #0x1\n"
            "mov x20, %x[width]\n"
            "mov x25, %x[in]\n"
            "ptrue p0.b\n"
            "mov x24, %x[outptr_raw]\n"
            "mov x23, #0x0\n"
            "lsr x10, x10, #0x1\n"
            "lsr x27, x27, #0x1\n"
            "mov x12, #0x0\n"
            "whilelt p2.s, x9, x20\n"
            "incw x9, ALL, MUL #1\n"
            "whilelt p3.s, x9, x20\n"
            "decw x9, ALL, MUL #1\n"
            "add x22, x25, x28, LSL #3\n"
            "1:"  // Width loop: Preamble: Loop
            "ldr x21, [x25], #0x8\n"
            "psel p4, p2, p12.s[w12,0]\n"
            "psel p5, p2, p11.s[w12,0]\n"
            "psel p6, p3, p12.s[w12,0]\n"
            "psel p7, p3, p11.s[w12,0]\n" 
            "ldr x20, [x22], #0x8\n"
            "ld1w  z20.s, p4/Z, [x21, x23, LSL #2]\n"
            "ld1w  z12.s, p5/Z, [x20, x23, LSL #2]\n" 
            "incw x23, ALL, MUL #1\n"
            "ld1w  z21.s, p6/Z, [x21, x23, LSL #2]\n"
            "ld1w  z13.s, p7/Z, [x20, x23, LSL #2]\n" 
            "decw x23, ALL, MUL #1\n"
            "BFCVT z0.h, p0/M, z20.s \n"
            "BFCVT z20.h, p0/M, z21.s \n"
            "UZP1 z20.h, z0.h, z20.h \n"
            "BFCVT z0.h, p0/M, z12.s \n"
            "BFCVT z12.h, p0/M, z13.s \n"
            "UZP1 z12.h, z0.h, z12.h \n"
            "mova za0h.s[w12,0], p0/M, z20.s\n"
            "mova za1h.s[w12,0], p0/M, z12.s\n"
            "add x12, x12, #0x1\n"
            "cmp x12, x28\n"
            "blt 1b\n"
            "incw x23, ALL, MUL #2\n"
            "incw x9, ALL, MUL #2\n"
            "cbz x10, 5f\n"
            "2:"  // Width loop
            "mov x20, %x[width]\n"
            "mov x25, %x[in]\n"
            "mov x12, #0x0\n"
            "whilelt p2.s, x9, x20\n"
            "incw x9, ALL, MUL #1\n"
            "whilelt p3.s, x9, x20\n"
            "decw x9, ALL, MUL #1\n"
            "add x22, x25, x28, LSL #3\n"
            "3:"  // Width loop: Odd: Loop
            "ldr x21, [x25], #0x8\n"
            "psel p4, p2, p12.s[w12,0]\n"
            "psel p5, p2, p11.s[w12,0]\n"
            "psel p6, p3, p12.s[w12,0]\n"
            "psel p7, p3, p11.s[w12,0]\n" 
            "mova z7.s, p0/M, za0v.s[w12,0]\n"
            "ldr x20, [x22], #0x8\n"
            "mova z15.s, p0/M, za1v.s[w12,0]\n"
            // "ld1w { z22.s-z23.s }, pn9/Z, [x21, x23, LSL #2]\n"
            // "ld1w { z26.s-z27.s }, pn8/Z, [x20, x23, LSL #2]\n"
            "ld1w  z22.s, p4/Z, [x21, x23, LSL #2]\n"
            "ld1w  z26.s, p5/Z, [x20, x23, LSL #2]\n" 
            "incw x23, ALL, MUL #1\n"
            "ld1w  z23.s, p6/Z, [x21, x23, LSL #2]\n"
            "ld1w  z27.s, p7/Z, [x20, x23, LSL #2]\n" 
            "decw x23, ALL, MUL #1\n"
            "st1w z7.s, p1, [x24] \n"
            "st1w z15.s, p1, [x24, #1, MUL VL] \n"
            "addvl x24, x24, #2\n"
            "BFCVT z0.h, p0/M, z22.s \n"
            "BFCVT z22.h, p0/M, z23.s \n"
            "UZP1 z22.h, z0.h, z22.h \n"
            "BFCVT z0.h, p0/M, z26.s \n"
            "BFCVT z26.h, p0/M, z27.s \n"
            "UZP1 z26.h, z0.h, z26.h \n"
            "mova za2h.s[w12,0], p0/M, z22.s\n"
            "mova za3h.s[w12,0], p0/M, z26.s\n"
            "add x12, x12, #0x1\n"
            "cmp x12, x28\n"
            "blt 3b\n"
            "incw x9, ALL, MUL #2\n"
            "mov x20, %x[width]\n"
            "mov x25, %x[in]\n"
            "incw x23, ALL, MUL #2\n"
            "mov x12, #0x0\n"
            "whilelt p2.s, x9, x20\n"
            "incw x9, ALL, MUL #1\n"
            "whilelt p3.s, x9, x20\n"
            "decw x9, ALL, MUL #1\n"
            "add x22, x25, x28, LSL #3\n"
            "4:"  // Width loop: Even: Loop
            "ldr x21, [x25], #0x8\n"
            "psel p4, p2, p12.s[w12,0]\n"
            "psel p5, p2, p11.s[w12,0]\n"
            "psel p6, p3, p12.s[w12,0]\n"
            "psel p7, p3, p11.s[w12,0]\n" 
            "mova z8.s, p0/M, za2v.s[w12,0]\n"
            "ldr x20, [x22], #0x8\n"
            "mova z9.s, p0/M, za3v.s[w12,0]\n"
            // "ld1w { z14.s-z15.s }, pn9/Z, [x21, x23, LSL #2]\n"
            // "ld1w { z12.s-z13.s }, pn8/Z, [x20, x23, LSL #2]\n"
            "ld1w  z14.s, p4/Z, [x21, x23, LSL #2]\n"
            "ld1w  z12.s, p5/Z, [x20, x23, LSL #2]\n" 
            "incw x23, ALL, MUL #1\n"
            "ld1w  z15.s, p6/Z, [x21, x23, LSL #2]\n"
            "ld1w  z13.s, p7/Z, [x20, x23, LSL #2]\n" 
            "decw x23, ALL, MUL #1\n"
            "st1w z8.s, p1, [x24] \n"
            "st1w z9.s, p1, [x24, #1, MUL VL] \n"
            "addvl x24, x24, #2\n"
            "BFCVT z0.h, p0/M, z14.s \n"
            "BFCVT z14.h, p0/M, z15.s \n"
            "UZP1 z14.h, z0.h, z14.h \n"
            "BFCVT z0.h, p0/M, z12.s \n"
            "BFCVT z12.h, p0/M, z13.s \n"
            "UZP1 z12.h, z0.h, z12.h \n"
            "mova za0h.s[w12,0], p0/M, z14.s\n"
            "mova za1h.s[w12,0], p0/M, z12.s\n"
            "add x12, x12, #0x1\n"
            "cmp x12, x28\n"
            "blt 4b\n"
            "subs x10, x10, #0x1\n"
            "incw x23, ALL, MUL #2\n"
            "incw x9, ALL, MUL #2\n"
            "bgt 2b\n"
            "5:"  // Width loop: Tails
            "cbnz x26, 8f\n"
            "mov x20, %x[width]\n"
            "mov x25, %x[in]\n"
            "mov x12, #0x0\n"
            "whilelt p2.s, x9, x20\n"
            "incw x9, ALL, MUL #1\n"
            "whilelt p3.s, x9, x20\n"
            "decw x9, ALL, MUL #1\n"
            "add x22, x25, x28, LSL #3\n"
            "6:"  // Width loop: Tails: Even: Odd: Loop
            "ldr x21, [x25], #0x8\n"
            "psel p4, p2, p12.s[w12,0]\n"
            "psel p5, p2, p11.s[w12,0]\n"
            "psel p6, p3, p12.s[w12,0]\n"
            "psel p7, p3, p11.s[w12,0]\n" 
            "mova z3.s, p0/M, za0v.s[w12,0]\n"
            "ldr x20, [x22], #0x8\n"
            "mova z11.s, p0/M, za1v.s[w12,0]\n"
            "ld1w  z12.s, p4/Z, [x21, x23, LSL #2]\n"
            "ld1w  z14.s, p5/Z, [x20, x23, LSL #2]\n" 
            "incw x23, ALL, MUL #1\n"
            "ld1w  z13.s, p6/Z, [x21, x23, LSL #2]\n"
            "ld1w  z15.s, p7/Z, [x20, x23, LSL #2]\n" 
            "decw x23, ALL, MUL #1\n"
            "st1w z3.s, p1, [x24] \n"
            "st1w z11.s, p1, [x24, #1, MUL VL] \n"
            "addvl x24, x24, #2\n"
            "BFCVT z0.h, p0/M, z12.s \n"
            "BFCVT z12.h, p0/M, z13.s \n"
            "UZP1 z12.h, z0.h, z12.h \n"
            "BFCVT z0.h, p0/M, z14.s \n"
            "BFCVT z14.h, p0/M, z15.s \n"
            "UZP1 z14.h, z0.h, z14.h \n"
            "mova za2h.s[w12,0], p0/M, z12.s\n"
            "mova za3h.s[w12,0], p0/M, z14.s\n"
            "add x12, x12, #0x1\n"
            "cmp x12, x28\n"
            "blt 6b\n"
            "mov x12, #0x0\n"
            "7:"  // Width loop: Tails: Even: Even: Loop
            "mova z14.s, p0/M, za2v.s[w12,0]\n"
            "mova z15.s, p0/M, za3v.s[w12,0]\n"
            "add x12, x12, #0x1\n"
            "cmp x12, x27\n" 
            "st1w z14.s, p1, [x24] \n" 
            "st1w z15.s, p1, [x24, #1, MUL VL] \n"
            "addvl x24, x24, #2\n"
            "blt 7b\n"
            "b 10f\n"
            "8:"  // Width loop: Tails: Odd
            "mov x12, #0x0\n"
            "9:"  // Width loop: Tails: Odd: Loop
            "mova z20.s, p0/M, za0v.s[w12,0]\n"
            "mova z21.s, p0/M, za1v.s[w12,0]\n"
            "add x12, x12, #0x1\n"
            "cmp x12, x27\n"
            "st1w z20.s, p1, [x24] \n"
            "st1w z21.s, p1, [x24, #1, MUL VL] \n"
            "addvl x24, x24, #2\n"
            "blt 9b\n"
            "10:"  // End
            "mov %x[outptr_raw], x24\n"
            ".inst 0xd503467f  // SMSTOP\n"
            : [outptr_raw] "+&r"(out)
            : [height] "r"(height), [in] "r"(in), [width] "r"(width)
            : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13",
              "p14", "p15", "x9", "x10", "x12", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0",
              "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16",
              "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31");
    }
}

#endif  // Architectural features check.
