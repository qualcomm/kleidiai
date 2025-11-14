//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
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

    KAI_ASM_CODE(matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme1_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme1_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme1_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme1_mopa)
    stp     x19, x20, [sp, -128 ]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     x25, x26, [sp, 48]
    stp     d8, d9,   [sp, 64]
    stp     d10, d11, [sp, 80]
    stp     d12, d13, [sp, 96]
    stp     d14, d15, [sp, 112]
    KAI_ASM_INST(0xd503477f)        // smstart
    cntw x14
    ptrue p0.b, all
    cntw x5                         // mr
    lsl x5, x5, #2
    whilelt p4.b, xzr, x5
    ldr x19, [x0, #0x10]           // rhs_packed
    KAI_ASM_INST(0x8558c009)       // ld1rw z9.s, p0/z, [x0, #0x60]  // min
    KAI_ASM_INST(0x8559c00a)       // ld1rw z10.s, p0/z, [x0, #0x64] //max
    fmov z11.h, #0.0
    ldr	x4, [x0, #0x50]            // bl
    ldr	x21, [x0, #0x18]           // stride
    ldr	x20, [x0]                  // dst
    mov x8, #0
    ldr x13, [x0, #0x40]           // n
    ldr x23, [x0, #0x48]           // k
    whilelt p2.h, x8, x13
    b.eq label_9                   // b.none label_9
KAI_ASM_LABEL(label_1)              // N loop
    ldr	x9, [x0, #0x38]            // m
    ldr	x22, [x0, #0x8]            // lhs_packed
    mov x24, x20
KAI_ASM_LABEL(label_2)              // M Loop
    mov x26, x19
    mov x3, x22
    cmp x9, x14
    csel x15, x9, x14, lo
    lsl x15, x15, #2
    ldr	x10, [x0, #0x48]           // k
    cmp x10, #0
    b.eq label_8
KAI_ASM_LABEL(label_3)              // K Loop
    zero {za}
    mov x11, x4
KAI_ASM_LABEL(label_4)              // BL Loop
    ld1w z2.s, p0/Z, [x26]
    ld1w z3.s, p0/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ld1h {z8.h}, p0/z, [x3]
    addvl x3, x3, #1
    lsr z7.b, z2.b, #4
    and z2.b, z2.b, #0x0F
    zip1 z4.b, z2.b, z7.b
    zip2 z5.b, z2.b, z7.b
    sub z4.b, z4.b, #8
    sub z5.b, z5.b, #8
    lsr z7.b, z3.b, #4
    and z3.b, z3.b, #0x0F
    zip1 z6.b, z3.b, z7.b
    zip2 z7.b, z3.b, z7.b
    sub z6.b, z6.b, #8
    sub z7.b, z7.b, #8
    smopa za0.s, p0/m, p0/m, z8.b, z4.b
    smopa za1.s, p0/m, p0/m, z8.b, z5.b
    smopa za2.s, p0/m, p0/m, z8.b, z6.b
    smopa za3.s, p0/m, p0/m, z8.b, z7.b
    subs x11, x11, #4
    b.gt label_4
    mov w12, #0
    mov x25, x24
    ld1b {z17.b}, p4/z, [x3]        // lhs sum
    ld1b {z16.b}, p4/z, [x3, #1, mul vl]  // lhs scale
    addvl x3, x3, #2
    ld1w z20.s, p0/Z, [x26]
    ld1w z21.s, p0/Z, [x26, #1, MUL VL]
    ld1w z22.s, p0/Z, [x26, #2, MUL VL]
    ld1w z23.s, p0/Z, [x26, #3, MUL VL]
    addvl x26, x26, #4
    ld1w z0.s, p0/Z, [x26]
    ld1w z1.s, p0/Z, [x26, #1, MUL VL]
    ld1w z2.s, p0/Z, [x26, #2, MUL VL]
    ld1w z3.s, p0/Z, [x26, #3, MUL VL]
    addvl x26, x26, #4
    pfalse p3.b
KAI_ASM_LABEL(label_5)
    pnext p3.s, p0, p3.s
    clastb z19.s, p3, z19.s, z16.s
    clastb z18.s, p3, z18.s, z17.s
    mova z28.b, p0/M, za0h.b[w12, 0]
    mova z29.b, p0/M, za0h.b[w12, 1]
    mova z30.b, p0/M, za0h.b[w12, 2]
    mova z31.b, p0/M, za0h.b[w12, 3]
    add w12, w12, #4
    fmul z4.s, z0.s, z19.s
    fmul z5.s, z1.s, z19.s
    fmul z6.s, z2.s, z19.s
    fmul z7.s, z3.s, z19.s
    scvtf z28.s, p0/M, z28.s
    scvtf z29.s, p0/M, z29.s
    scvtf z30.s, p0/M, z30.s
    scvtf z31.s, p0/M, z31.s
    inch x8, all
    whilelt p5.h, x8, x13
    dech x8, all
    cmp x10, x23
    b.ne label_6
    fmul z24.s, z20.s, z18.s
    fmul z25.s, z21.s, z18.s
    fmul z26.s, z22.s, z18.s
    fmul z27.s, z23.s, z18.s
    fmla z24.s, p0/m,  z4.s, z28.s
    fmla z25.s, p0/m,  z5.s, z29.s
    fmla z26.s, p0/m,  z6.s, z30.s
    fmla z27.s, p0/m,  z7.s, z31.s
    b label_7
KAI_ASM_LABEL(label_6)
    ld1h z12.h, p2/Z, [x25]
    ld1h z13.h, p5/Z, [x25, #1, MUL VL]
    zip1 z24.h, z12.h, z11.h
    zip2 z25.h, z12.h, z11.h
    zip1 z26.h, z13.h, z11.h
    zip2 z27.h, z13.h, z11.h
    fcvt z24.s, p0/m, z24.h
    fcvt z25.s, p0/m, z25.h
    fcvt z26.s, p0/m, z26.h
    fcvt z27.s, p0/m, z27.h
    fmla z24.s, p0/m, z20.s, z18.s
    fmla z25.s, p0/m, z21.s, z18.s
    fmla z26.s, p0/m, z22.s, z18.s
    fmla z27.s, p0/m, z23.s, z18.s
    fmla z24.s, p0/m,  z4.s, z28.s
    fmla z25.s, p0/m,  z5.s, z29.s
    fmla z26.s, p0/m,  z6.s, z30.s
    fmla z27.s, p0/m,  z7.s, z31.s
KAI_ASM_LABEL(label_7)
    fcvt  z24.h, p0/M, z24.s
    fcvt  z28.h, p0/M, z25.s
    uzp1  z28.h, z24.h, z28.h
    fcvt  z26.h, p0/M, z26.s
    fcvt  z29.h, p0/M, z27.s
    uzp1  z29.h, z26.h, z29.h   
    st1h z28.h, p2, [x25]
    st1h z29.h, p5, [x25, #1, MUL VL]
    add x25, x25, x21
    cmp x12, x15
    blt label_5
    subs x10, x10, x4
    b.gt label_3
KAI_ASM_LABEL(label_8)
    ldr	x5, [x0, #0x30]
    add x5, x5, x19
    ld1w z12.s, p0/Z, [x5]
    ld1w z13.s, p0/Z, [x5, #1, MUL VL]
    ld1w z14.s, p0/Z, [x5, #2, MUL VL]
    ld1w z15.s, p0/Z, [x5, #3, MUL VL]
    mov x12, 0
KAI_ASM_LABEL(label_10)
    ld1h z0.h, p2/Z, [x24]
    ld1h z1.h, p5/Z, [x24, #1, MUL VL]
    zip1 z24.h, z0.h, z11.h
    zip2 z25.h, z0.h, z11.h
    zip1 z26.h, z1.h, z11.h
    zip2 z27.h, z1.h, z11.h
    fcvt z24.s, p0/m, z24.h
    fcvt z25.s, p0/m, z25.h
    fcvt z26.s, p0/m, z26.h
    fcvt z27.s, p0/m, z27.h
    fadd z24.s, p0/m, z24.s, z12.s
    fadd z25.s, p0/m, z25.s, z13.s
    fadd z26.s, p0/m, z26.s, z14.s
    fadd z27.s, p0/m, z27.s, z15.s
    FMIN z24.s, p0/M, z24.s, z10.s
    FMIN z25.s, p0/M, z25.s, z10.s
    FMIN z26.s, p0/M, z26.s, z10.s
    FMIN z27.s, p0/M, z27.s, z10.s
    FMAX z24.s, p0/M, z24.s, z9.s
    FMAX z25.s, p0/M, z25.s, z9.s
    FMAX z26.s, p0/M, z26.s, z9.s
    FMAX z27.s, p0/M, z27.s, z9.s
    fcvt  z24.h, p0/M, z24.s
    fcvt  z28.h, p0/M, z25.s
    uzp1  z28.h, z24.h, z28.h
    fcvt  z26.h, p0/M, z26.s
    fcvt  z29.h, p0/M, z27.s
    uzp1  z29.h, z26.h, z29.h
    st1h z28.h, p2, [x24]
    st1h z29.h, p5, [x24, #1, MUL VL]
    add x24, x24, x21
    add x12, x12, #4
    cmp x12, x15
    blt label_10
    ldr	x5, [x0, #0x20]
    add x22, x22, x5
    mov x24, x25
    decw x9, all
    cmp x9, #0
    b.gt label_2
    incb x20, all, mul #2
    ldr	x5, [x0, #0x28]
    add x19, x19, x5
    incb x8, all
    whilelt p2.h, x8, x13
    b.mi label_1                    // b.first label_1
KAI_ASM_LABEL(label_9)
    KAI_ASM_INST(0xd503467f)        // smstop
    ldp     d14, d15, [sp, 112]
    ldp     d12, d13, [sp, 96]
    ldp     d10, d11, [sp, 80]
    ldp     d8, d9,   [sp, 64]
    ldp     x25, x26, [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp],128
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme1_mopa)

    KAI_ASM_END
