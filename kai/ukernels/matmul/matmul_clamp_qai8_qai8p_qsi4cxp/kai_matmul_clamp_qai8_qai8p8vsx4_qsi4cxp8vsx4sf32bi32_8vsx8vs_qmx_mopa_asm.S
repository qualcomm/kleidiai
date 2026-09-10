//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//

#if defined(_MSC_VER)
    #define KAI_ASM_GLOBAL(name) GLOBAL name
    #define KAI_ASM_FUNCTION_TYPE(name)
    #define KAI_ASM_FUNCTION_LABEL(name) name PROC
    #define KAI_ASM_FUNCTION_END(name) ENDP

    #define KAI_ASM_CODE(name) AREA name, CODE, READONLY
    #define KAI_ASM_ALIGN
    #define KAI_ASM_LABEL(name) name
    #define KAI_ASM_INST(hex) DCD hex
    #define KAI_ASM_END END
#else
    #if defined(__APPLE__)
        #define KAI_ASM_GLOBAL(name) .globl _##name
        #define KAI_ASM_FUNCTION_TYPE(name)
        #define KAI_ASM_FUNCTION_LABEL(name) _##name:
        #define KAI_ASM_FUNCTION_END(name)
    #else
        #define KAI_ASM_GLOBAL(name) .global name
        #define KAI_ASM_FUNCTION_TYPE(name) .type name, %function
        #define KAI_ASM_FUNCTION_LABEL(name) name:
        #define KAI_ASM_FUNCTION_END(name) .size name, .-name
    #endif

    #define KAI_ASM_CODE(name) .text
    #define KAI_ASM_ALIGN .p2align 4,,11
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_qmx_mopa)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    str x19, [sp, 136]              // callee-saved; holds the scale-region offset (see below)
    KAI_ASM_INST(0xd503477f)        // smstart
    // QMX (SME1) port of the sme2_mopa sibling. No zt0/luti4 is loaded: the i4 -> i8
    // expansion is arithmetic (see label_4). args.lut_ptr is unused and left NULL.
    mov x15, #0x0
    ldr x14, [x0, #0x30]            // K
    ptrue p1.b
    ldr w13, [x0, #0x20]            // M
    mov x11, #0x0
    ldr w10, [x0, #0x28]            // N
    add x14, x14, #0x3
    ldr x9, [x0, #0x0]              // lhs
    lsr x14, x14, #0x2              // x14 = ceil(K/4) = number of kr=4 RHS slices
    // x19 = byte offset from an N-panel's base to its scale region:
    //   nr*4 (bias) + roundup(K,32)*nr/2 (payload) == (2 + roundup(K,32)/4) vectors.
    // Computing it removes the sme2 kernel's 8-way "skip the unused slices" jump table.
    cntb x2
    ldr x3, [x0, #0x30]
    add x3, x3, #31
    lsr x3, x3, #5
    lsl x3, x3, #3                  // roundup(K,32)/4
    add x3, x3, #2
    mul x19, x3, x2
KAI_ASM_LABEL(label_1)  // M loop
    ldr x7, [x0, #0x8]              // rhs block base
KAI_ASM_LABEL(label_2)  // N loop
    mov x28, x7
    ld1w z2.s, p1/z, [x28]                  // bias, N half 0
    ld1w z3.s, p1/z, [x28, #1, MUL VL]      // bias, N half 1
    KAI_ASM_INST(0xc00800ff)        // zero { za }
    mov x27, x9                     // lhs
    addvl x28, x28, #2
    addha za0.s, p1/m, p1/m, z2.s
    addha za1.s, p1/m, p1/m, z3.s
    addha za2.s, p1/m, p1/m, z2.s
    addha za3.s, p1/m, p1/m, z3.s
    lsr x21, x14, #0x2              // groups of 4 slices
    and x20, x14, #0x3              // leftover single slices
    and x6, x14, #0x7               // used to skip the 32-K block's unused slices
    cbz x21, label_6
    subs x21, x21, #0x1
    ld1b z0.b, p1/z, [x27]
    ld1b z1.b, p1/z, [x27, #1, MUL VL]
    ld1b z2.b, p1/z, [x27, #2, MUL VL]
    ld1b z3.b, p1/z, [x27, #3, MUL VL]
    ld1b z4.b, p1/z, [x27, #4, MUL VL]
    ld1b z5.b, p1/z, [x27, #5, MUL VL]
    ld1b z6.b, p1/z, [x27, #6, MUL VL]
    ld1b z7.b, p1/z, [x27, #7, MUL VL]
    addvl x27, x27, #8
    ld1b z8.b, p1/z, [x28]
    ld1b z9.b, p1/z, [x28, #1, MUL VL]
    ld1b z10.b, p1/z, [x28, #2, MUL VL]
    ld1b z11.b, p1/z, [x28, #3, MUL VL]
    addvl x28, x28, #4
    // zip-free i4 -> i8 decode (s4s0 payload): 3 ops per vector yield two SMOPA operands.
    // Vector t covers nr/2 columns; plane A is K[k0..k0+3], plane B is K[k0+4..k0+7].
    lsl z16.b, z8.b, #4
    asr z16.b, z16.b, #4          // plane A = K[k0..k0+3]
    asr z17.b, z8.b, #4        // plane B = K[k0+4..k0+7]
    lsl z18.b, z9.b, #4
    asr z18.b, z18.b, #4          // plane A = K[k0..k0+3]
    asr z19.b, z9.b, #4        // plane B = K[k0+4..k0+7]
    lsl z20.b, z10.b, #4
    asr z20.b, z20.b, #4          // plane A = K[k0..k0+3]
    asr z21.b, z10.b, #4        // plane B = K[k0+4..k0+7]
    lsl z22.b, z11.b, #4
    asr z22.b, z22.b, #4          // plane A = K[k0..k0+3]
    asr z23.b, z11.b, #4        // plane B = K[k0+4..k0+7]
    ble label_5
KAI_ASM_LABEL(label_4)  // K loop: 4 slices (16 K) per iteration
    // Next iteration's loads interleave with this iteration's smopa; each LHS pair is
    // reloaded only after the smopa that consumes it.
    smopa za0.s, p1/m, p1/m, z0.b, z16.b
    smopa za1.s, p1/m, p1/m, z0.b, z18.b
    smopa za2.s, p1/m, p1/m, z1.b, z16.b
    smopa za3.s, p1/m, p1/m, z1.b, z18.b
    smopa za0.s, p1/m, p1/m, z2.b, z17.b
    smopa za1.s, p1/m, p1/m, z2.b, z19.b
    smopa za2.s, p1/m, p1/m, z3.b, z17.b
    smopa za3.s, p1/m, p1/m, z3.b, z19.b
    ld1b z0.b, p1/z, [x27]
    ld1b z1.b, p1/z, [x27, #1, MUL VL]
    ld1b z2.b, p1/z, [x27, #2, MUL VL]
    ld1b z3.b, p1/z, [x27, #3, MUL VL]
    smopa za0.s, p1/m, p1/m, z4.b, z20.b
    smopa za1.s, p1/m, p1/m, z4.b, z22.b
    smopa za2.s, p1/m, p1/m, z5.b, z20.b
    smopa za3.s, p1/m, p1/m, z5.b, z22.b
    smopa za0.s, p1/m, p1/m, z6.b, z21.b
    smopa za1.s, p1/m, p1/m, z6.b, z23.b
    smopa za2.s, p1/m, p1/m, z7.b, z21.b
    smopa za3.s, p1/m, p1/m, z7.b, z23.b
    ld1b z4.b, p1/z, [x27, #4, MUL VL]
    ld1b z5.b, p1/z, [x27, #5, MUL VL]
    ld1b z6.b, p1/z, [x27, #6, MUL VL]
    ld1b z7.b, p1/z, [x27, #7, MUL VL]
    addvl x27, x27, #8
    ld1b z8.b, p1/z, [x28]
    ld1b z9.b, p1/z, [x28, #1, MUL VL]
    ld1b z10.b, p1/z, [x28, #2, MUL VL]
    ld1b z11.b, p1/z, [x28, #3, MUL VL]
    addvl x28, x28, #4
    subs x21, x21, #0x1
    lsl z16.b, z8.b, #4
    asr z16.b, z16.b, #4          // plane A = K[k0..k0+3]
    asr z17.b, z8.b, #4        // plane B = K[k0+4..k0+7]
    lsl z18.b, z9.b, #4
    asr z18.b, z18.b, #4          // plane A = K[k0..k0+3]
    asr z19.b, z9.b, #4        // plane B = K[k0+4..k0+7]
    lsl z20.b, z10.b, #4
    asr z20.b, z20.b, #4          // plane A = K[k0..k0+3]
    asr z21.b, z10.b, #4        // plane B = K[k0+4..k0+7]
    lsl z22.b, z11.b, #4
    asr z22.b, z22.b, #4          // plane A = K[k0..k0+3]
    asr z23.b, z11.b, #4        // plane B = K[k0+4..k0+7]
    bgt label_4
KAI_ASM_LABEL(label_5)  // K loop: tail
    smopa za0.s, p1/m, p1/m, z0.b, z16.b
    smopa za1.s, p1/m, p1/m, z0.b, z18.b
    smopa za2.s, p1/m, p1/m, z1.b, z16.b
    smopa za3.s, p1/m, p1/m, z1.b, z18.b
    smopa za0.s, p1/m, p1/m, z2.b, z17.b
    smopa za1.s, p1/m, p1/m, z2.b, z19.b
    smopa za2.s, p1/m, p1/m, z3.b, z17.b
    smopa za3.s, p1/m, p1/m, z3.b, z19.b
    smopa za0.s, p1/m, p1/m, z4.b, z20.b
    smopa za1.s, p1/m, p1/m, z4.b, z22.b
    smopa za2.s, p1/m, p1/m, z5.b, z20.b
    smopa za3.s, p1/m, p1/m, z5.b, z22.b
    smopa za0.s, p1/m, p1/m, z6.b, z21.b
    smopa za1.s, p1/m, p1/m, z6.b, z23.b
    smopa za2.s, p1/m, p1/m, z7.b, z21.b
    smopa za3.s, p1/m, p1/m, z7.b, z23.b
KAI_ASM_LABEL(label_6)  // K oddments: 1..3 leftover 4-K slices
    cbz x20, label_8
    // Slices 0 and 1 are planes A and B of the SAME pair of vectors, so only slice 2
    // needs a second pair loaded. RHS advances 2 vectors for 1-2 slices, 4 for 3.
    ld1b z8.b, p1/z, [x28]
    ld1b z9.b, p1/z, [x28, #1, MUL VL]
    lsl z16.b, z8.b, #4
    asr z16.b, z16.b, #4          // plane A = K[k0..k0+3]
    asr z17.b, z8.b, #4        // plane B = K[k0+4..k0+7]
    lsl z18.b, z9.b, #4
    asr z18.b, z18.b, #4          // plane A = K[k0..k0+3]
    asr z19.b, z9.b, #4        // plane B = K[k0+4..k0+7]
    ld1b z0.b, p1/z, [x27]
    ld1b z1.b, p1/z, [x27, #1, MUL VL]
    smopa za0.s, p1/m, p1/m, z0.b, z16.b
    smopa za1.s, p1/m, p1/m, z0.b, z18.b
    smopa za2.s, p1/m, p1/m, z1.b, z16.b
    smopa za3.s, p1/m, p1/m, z1.b, z18.b
    cmp x20, #0x1
    b.eq label_17
    ld1b z2.b, p1/z, [x27, #2, MUL VL]
    ld1b z3.b, p1/z, [x27, #3, MUL VL]
    smopa za0.s, p1/m, p1/m, z2.b, z17.b
    smopa za1.s, p1/m, p1/m, z2.b, z19.b
    smopa za2.s, p1/m, p1/m, z3.b, z17.b
    smopa za3.s, p1/m, p1/m, z3.b, z19.b
    cmp x20, #0x2
    b.eq label_18
    ld1b z10.b, p1/z, [x28, #2, MUL VL]
    ld1b z11.b, p1/z, [x28, #3, MUL VL]
    lsl z20.b, z10.b, #4
    asr z20.b, z20.b, #4          // plane A = K[k0..k0+3]
    asr z21.b, z10.b, #4        // plane B = K[k0+4..k0+7]
    lsl z22.b, z11.b, #4
    asr z22.b, z22.b, #4          // plane A = K[k0..k0+3]
    asr z23.b, z11.b, #4        // plane B = K[k0+4..k0+7]
    ld1b z4.b, p1/z, [x27, #4, MUL VL]
    ld1b z5.b, p1/z, [x27, #5, MUL VL]
    smopa za0.s, p1/m, p1/m, z4.b, z20.b
    smopa za1.s, p1/m, p1/m, z4.b, z22.b
    smopa za2.s, p1/m, p1/m, z5.b, z20.b
    smopa za3.s, p1/m, p1/m, z5.b, z22.b
    addvl x27, x27, #6
    addvl x28, x28, #4
    b label_8
KAI_ASM_LABEL(label_17)  // K oddments: one slice
    addvl x27, x27, #2
    addvl x28, x28, #2
    b label_8
KAI_ASM_LABEL(label_18)  // K oddments: two slices
    addvl x27, x27, #4
    addvl x28, x28, #2
KAI_ASM_LABEL(label_8)  // K oddments: end
    // Jump straight to this panel's scale region (see x19 above).
    add x28, x7, x19
KAI_ASM_LABEL(label_20)  // Store: setup
    ldr x26, [x0, #0x10]            // dst
    sub x25, x13, x15               // rows left
    cntw x24                        // W = .s elements per vector
    ld1rw {z26.s}, p1/z, [x0, #56]  // min
    ldr x23, [x0, #0x18]            // ldcb
    whilelt p0.h, x11, x10
    cmp x25, x24
    ld1rw {z23.s}, p1/z, [x0, #60]  // max
    csel x22, x25, x24, LT
    ld1rw {z0.s}, p1/z, [x0, #64]   // result_zero_point
    mov x12, #0x0
    add x26, x26, x11               // dst += n
    lsr x21, x22, #0x2
    ld1w z2.s, p1/z, [x28]                  // scale, N half 0
    ld1w z3.s, p1/z, [x28, #1, MUL VL]      // scale, N half 1
    madd x26, x15, x23, x26         // dst += m * ldc
    addvl x28, x28, #2
    and x20, x22, #0x3
    cbz x21, label_11
KAI_ASM_LABEL(label_10)  // Store: za0/za1, 4 rows per iteration
    mova z8.s, p1/m, za0h.s[w12, 0]
    mova z9.s, p1/m, za0h.s[w12, 1]
    mova z10.s, p1/m, za0h.s[w12, 2]
    mova z11.s, p1/m, za0h.s[w12, 3]
    mova z16.s, p1/m, za1h.s[w12, 0]
    mova z17.s, p1/m, za1h.s[w12, 1]
    mova z18.s, p1/m, za1h.s[w12, 2]
    mova z19.s, p1/m, za1h.s[w12, 3]
    scvtf z8.s, p1/m, z8.s
    scvtf z9.s, p1/m, z9.s
    scvtf z10.s, p1/m, z10.s
    scvtf z11.s, p1/m, z11.s
    scvtf z16.s, p1/m, z16.s
    scvtf z17.s, p1/m, z17.s
    scvtf z18.s, p1/m, z18.s
    scvtf z19.s, p1/m, z19.s
    fmul z8.s, z8.s, z2.s
    fmul z9.s, z9.s, z2.s
    fmul z10.s, z10.s, z2.s
    fmul z11.s, z11.s, z2.s
    fmul z16.s, z16.s, z3.s
    fmul z17.s, z17.s, z3.s
    fmul z18.s, z18.s, z3.s
    fmul z19.s, z19.s, z3.s
    add x12, x12, #0x4
    cmp x12, x21, LSL #2
    frintn z8.s, p1/m, z8.s
    frintn z9.s, p1/m, z9.s
    frintn z10.s, p1/m, z10.s
    frintn z11.s, p1/m, z11.s
    frintn z16.s, p1/m, z16.s
    frintn z17.s, p1/m, z17.s
    frintn z18.s, p1/m, z18.s
    frintn z19.s, p1/m, z19.s
    fcvtzs z8.s, p1/m, z8.s
    fcvtzs z9.s, p1/m, z9.s
    fcvtzs z10.s, p1/m, z10.s
    fcvtzs z11.s, p1/m, z11.s
    fcvtzs z16.s, p1/m, z16.s
    fcvtzs z17.s, p1/m, z17.s
    fcvtzs z18.s, p1/m, z18.s
    fcvtzs z19.s, p1/m, z19.s
    add z8.s, z8.s, z0.s
    add z9.s, z9.s, z0.s
    add z10.s, z10.s, z0.s
    add z11.s, z11.s, z0.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z0.s
    add z18.s, z18.s, z0.s
    add z19.s, z19.s, z0.s
    sclamp z8.s, z26.s, z23.s
    sclamp z9.s, z26.s, z23.s
    sclamp z10.s, z26.s, z23.s
    sclamp z11.s, z26.s, z23.s
    sclamp z16.s, z26.s, z23.s
    sclamp z17.s, z26.s, z23.s
    sclamp z18.s, z26.s, z23.s
    sclamp z19.s, z26.s, z23.s
    uzp1 z4.h, z8.h, z16.h
    uzp1 z5.h, z9.h, z17.h
    uzp1 z6.h, z10.h, z18.h
    uzp1 z7.h, z11.h, z19.h
    st1b {z4.h}, p0, [x26]
    add x26, x26, x23
    st1b {z5.h}, p0, [x26]
    add x26, x26, x23
    st1b {z6.h}, p0, [x26]
    add x26, x26, x23
    st1b {z7.h}, p0, [x26]
    add x26, x26, x23
    blt label_10
KAI_ASM_LABEL(label_11)  // Store: za0/za1, row oddments
    cbz x20, label_12
    mova z8.s, p1/m, za0h.s[w12, 0]
    mova z9.s, p1/m, za0h.s[w12, 1]
    mova z10.s, p1/m, za0h.s[w12, 2]
    mova z11.s, p1/m, za0h.s[w12, 3]
    mova z12.s, p1/m, za1h.s[w12, 0]
    mova z13.s, p1/m, za1h.s[w12, 1]
    mova z14.s, p1/m, za1h.s[w12, 2]
    mova z15.s, p1/m, za1h.s[w12, 3]
    scvtf z8.s, p1/m, z8.s
    scvtf z9.s, p1/m, z9.s
    scvtf z10.s, p1/m, z10.s
    scvtf z11.s, p1/m, z11.s
    scvtf z12.s, p1/m, z12.s
    scvtf z13.s, p1/m, z13.s
    scvtf z14.s, p1/m, z14.s
    scvtf z15.s, p1/m, z15.s
    fmul z8.s, z8.s, z2.s
    fmul z9.s, z9.s, z2.s
    fmul z10.s, z10.s, z2.s
    fmul z11.s, z11.s, z2.s
    fmul z12.s, z12.s, z3.s
    fmul z13.s, z13.s, z3.s
    fmul z14.s, z14.s, z3.s
    fmul z15.s, z15.s, z3.s
    subs x20, x20, #0x1
    frintn z8.s, p1/m, z8.s
    frintn z9.s, p1/m, z9.s
    frintn z10.s, p1/m, z10.s
    frintn z11.s, p1/m, z11.s
    frintn z12.s, p1/m, z12.s
    frintn z13.s, p1/m, z13.s
    frintn z14.s, p1/m, z14.s
    frintn z15.s, p1/m, z15.s
    fcvtzs z8.s, p1/m, z8.s
    fcvtzs z9.s, p1/m, z9.s
    fcvtzs z10.s, p1/m, z10.s
    fcvtzs z11.s, p1/m, z11.s
    fcvtzs z12.s, p1/m, z12.s
    fcvtzs z13.s, p1/m, z13.s
    fcvtzs z14.s, p1/m, z14.s
    fcvtzs z15.s, p1/m, z15.s
    add z8.s, z8.s, z0.s
    add z9.s, z9.s, z0.s
    add z10.s, z10.s, z0.s
    add z11.s, z11.s, z0.s
    add z12.s, z12.s, z0.s
    add z13.s, z13.s, z0.s
    add z14.s, z14.s, z0.s
    add z15.s, z15.s, z0.s
    sclamp z8.s, z26.s, z23.s
    sclamp z9.s, z26.s, z23.s
    sclamp z10.s, z26.s, z23.s
    sclamp z11.s, z26.s, z23.s
    sclamp z12.s, z26.s, z23.s
    sclamp z13.s, z26.s, z23.s
    sclamp z14.s, z26.s, z23.s
    sclamp z15.s, z26.s, z23.s
    uzp1 z20.h, z8.h, z12.h
    st1b {z20.h}, p0, [x26]
    add x26, x26, x23
    beq label_12
    subs x20, x20, #0x1
    uzp1 z21.h, z9.h, z13.h
    st1b {z21.h}, p0, [x26]
    add x26, x26, x23
    beq label_12
    subs x20, x20, #0x1
    uzp1 z22.h, z10.h, z14.h
    st1b {z22.h}, p0, [x26]
    add x26, x26, x23
KAI_ASM_LABEL(label_12)  // Store: M half 0 done
    subs x25, x25, x22
    beq label_16
    cmp x25, x24
    mov x12, #0x0
    csel x20, x25, x24, LT
    lsr x21, x20, #0x2
    and x20, x20, #0x3
    cbz x21, label_14
KAI_ASM_LABEL(label_13)  // Store: za2/za3, 4 rows per iteration
    mova z8.s, p1/m, za2h.s[w12, 0]
    mova z9.s, p1/m, za2h.s[w12, 1]
    mova z10.s, p1/m, za2h.s[w12, 2]
    mova z11.s, p1/m, za2h.s[w12, 3]
    mova z16.s, p1/m, za3h.s[w12, 0]
    mova z17.s, p1/m, za3h.s[w12, 1]
    mova z18.s, p1/m, za3h.s[w12, 2]
    mova z19.s, p1/m, za3h.s[w12, 3]
    scvtf z8.s, p1/m, z8.s
    scvtf z9.s, p1/m, z9.s
    scvtf z10.s, p1/m, z10.s
    scvtf z11.s, p1/m, z11.s
    scvtf z16.s, p1/m, z16.s
    scvtf z17.s, p1/m, z17.s
    scvtf z18.s, p1/m, z18.s
    scvtf z19.s, p1/m, z19.s
    fmul z8.s, z8.s, z2.s
    fmul z9.s, z9.s, z2.s
    fmul z10.s, z10.s, z2.s
    fmul z11.s, z11.s, z2.s
    fmul z16.s, z16.s, z3.s
    fmul z17.s, z17.s, z3.s
    fmul z18.s, z18.s, z3.s
    fmul z19.s, z19.s, z3.s
    add x12, x12, #0x4
    cmp x12, x21, LSL #2
    frintn z8.s, p1/m, z8.s
    frintn z9.s, p1/m, z9.s
    frintn z10.s, p1/m, z10.s
    frintn z11.s, p1/m, z11.s
    frintn z16.s, p1/m, z16.s
    frintn z17.s, p1/m, z17.s
    frintn z18.s, p1/m, z18.s
    frintn z19.s, p1/m, z19.s
    fcvtzs z8.s, p1/m, z8.s
    fcvtzs z9.s, p1/m, z9.s
    fcvtzs z10.s, p1/m, z10.s
    fcvtzs z11.s, p1/m, z11.s
    fcvtzs z16.s, p1/m, z16.s
    fcvtzs z17.s, p1/m, z17.s
    fcvtzs z18.s, p1/m, z18.s
    fcvtzs z19.s, p1/m, z19.s
    add z8.s, z8.s, z0.s
    add z9.s, z9.s, z0.s
    add z10.s, z10.s, z0.s
    add z11.s, z11.s, z0.s
    add z16.s, z16.s, z0.s
    add z17.s, z17.s, z0.s
    add z18.s, z18.s, z0.s
    add z19.s, z19.s, z0.s
    sclamp z8.s, z26.s, z23.s
    sclamp z9.s, z26.s, z23.s
    sclamp z10.s, z26.s, z23.s
    sclamp z11.s, z26.s, z23.s
    sclamp z16.s, z26.s, z23.s
    sclamp z17.s, z26.s, z23.s
    sclamp z18.s, z26.s, z23.s
    sclamp z19.s, z26.s, z23.s
    uzp1 z4.h, z8.h, z16.h
    uzp1 z5.h, z9.h, z17.h
    uzp1 z6.h, z10.h, z18.h
    uzp1 z7.h, z11.h, z19.h
    st1b {z4.h}, p0, [x26]
    add x26, x26, x23
    st1b {z5.h}, p0, [x26]
    add x26, x26, x23
    st1b {z6.h}, p0, [x26]
    add x26, x26, x23
    st1b {z7.h}, p0, [x26]
    add x26, x26, x23
    blt label_13
KAI_ASM_LABEL(label_14)  // Store: za2/za3, row oddments
    cbz x20, label_16
    mova z8.s, p1/m, za2h.s[w12, 0]
    mova z9.s, p1/m, za2h.s[w12, 1]
    mova z10.s, p1/m, za2h.s[w12, 2]
    mova z11.s, p1/m, za2h.s[w12, 3]
    mova z12.s, p1/m, za3h.s[w12, 0]
    mova z13.s, p1/m, za3h.s[w12, 1]
    mova z14.s, p1/m, za3h.s[w12, 2]
    mova z15.s, p1/m, za3h.s[w12, 3]
    scvtf z8.s, p1/m, z8.s
    scvtf z9.s, p1/m, z9.s
    scvtf z10.s, p1/m, z10.s
    scvtf z11.s, p1/m, z11.s
    scvtf z12.s, p1/m, z12.s
    scvtf z13.s, p1/m, z13.s
    scvtf z14.s, p1/m, z14.s
    scvtf z15.s, p1/m, z15.s
    fmul z8.s, z8.s, z2.s
    fmul z9.s, z9.s, z2.s
    fmul z10.s, z10.s, z2.s
    fmul z11.s, z11.s, z2.s
    fmul z12.s, z12.s, z3.s
    fmul z13.s, z13.s, z3.s
    fmul z14.s, z14.s, z3.s
    fmul z15.s, z15.s, z3.s
    subs x20, x20, #0x1
    frintn z8.s, p1/m, z8.s
    frintn z9.s, p1/m, z9.s
    frintn z10.s, p1/m, z10.s
    frintn z11.s, p1/m, z11.s
    frintn z12.s, p1/m, z12.s
    frintn z13.s, p1/m, z13.s
    frintn z14.s, p1/m, z14.s
    frintn z15.s, p1/m, z15.s
    fcvtzs z8.s, p1/m, z8.s
    fcvtzs z9.s, p1/m, z9.s
    fcvtzs z10.s, p1/m, z10.s
    fcvtzs z11.s, p1/m, z11.s
    fcvtzs z12.s, p1/m, z12.s
    fcvtzs z13.s, p1/m, z13.s
    fcvtzs z14.s, p1/m, z14.s
    fcvtzs z15.s, p1/m, z15.s
    add z8.s, z8.s, z0.s
    add z9.s, z9.s, z0.s
    add z10.s, z10.s, z0.s
    add z11.s, z11.s, z0.s
    add z12.s, z12.s, z0.s
    add z13.s, z13.s, z0.s
    add z14.s, z14.s, z0.s
    add z15.s, z15.s, z0.s
    sclamp z8.s, z26.s, z23.s
    sclamp z9.s, z26.s, z23.s
    sclamp z10.s, z26.s, z23.s
    sclamp z11.s, z26.s, z23.s
    sclamp z12.s, z26.s, z23.s
    sclamp z13.s, z26.s, z23.s
    sclamp z14.s, z26.s, z23.s
    sclamp z15.s, z26.s, z23.s
    uzp1 z20.h, z8.h, z12.h
    st1b {z20.h}, p0, [x26]
    add x26, x26, x23
    beq label_16
    subs x20, x20, #0x1
    uzp1 z21.h, z9.h, z13.h
    st1b {z21.h}, p0, [x26]
    add x26, x26, x23
    beq label_16
    subs x20, x20, #0x1
    uzp1 z22.h, z10.h, z14.h
    st1b {z22.h}, p0, [x26]
    add x26, x26, x23
KAI_ASM_LABEL(label_16)  // Store: end
    mov x7, x28
    incw x11, ALL, MUL #2
    cmp x11, x10
    blt label_2
    incw x15, ALL, MUL #2
    mov x11, #0x0
    cmp x15, x13
    mov x9, x27
    blt label_1
    KAI_ASM_INST(0xd503467f)        // smstop
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldr x19, [sp, 136]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_qmx_mopa)

    KAI_ASM_END
