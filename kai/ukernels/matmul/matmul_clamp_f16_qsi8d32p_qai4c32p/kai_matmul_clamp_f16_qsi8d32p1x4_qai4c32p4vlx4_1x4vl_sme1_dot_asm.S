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

    KAI_ASM_CODE(matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme1_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme1_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme1_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme1_dot)
    stp     x19, x20, [sp, -112 ]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     d8, d9,   [sp, 48]
    stp     d10, d11, [sp, 64]
    stp     d12, d13, [sp, 80]
    stp     d14, d15, [sp, 96]
    KAI_ASM_INST(0xd503477f)    // smstart
    ptrue p2.b, all

    ldr	x10, [x0, #0x10]        // rhs_packed
    ldr	x5, [x0]                // dst
    KAI_ASM_INST(0x8550c81e)    // ld1rw z30.s, p2/z, [x0, #0x40] // min
    KAI_ASM_INST(0x8551c81f)    // ld1rw z31.s, p2/z, [x0, #0x44] // max
    mov x4, #0
    ldr x24, [x0, #0x20]        // n
    whilelt p4.h, x4, x24
    ldr	x19, [x0, #0x28]        // k
    ldr	x20, [x0, #0x30]        // bl
    b.eq label_5                // b.none  label_5
KAI_ASM_LABEL(label_1)          // N LOOP
    ldr x21, [x0, #0x8]         // lhs_packed
    mov x23, x10
    dup z24.s, #0
    dup z25.s, #0
    dup z26.s, #0
    dup z27.s, #0
    mov w8, #0
    mov x6, #0
    whilelt p1.s, x6, x19
    b.eq label_4                // b.none label_4
KAI_ASM_LABEL(label_2)          // K Loop
    zero {za}
    dup z16.s, #0 
    dup z17.s, #0 
    dup z18.s, #0 
    dup z19.s, #0
    mov x13, x20
KAI_ASM_LABEL(label_3)          // Bl Loop
    ld1rqb  { z0.b }, p2/z , [x21]
    add x21, x21, #16
    ld1h z12.h, p2/Z, [x23]
    ld1h z13.h, p2/Z, [x23, #1, MUL VL]
    ld1h z14.h, p2/Z, [x23, #2, MUL VL]
    ld1h z15.h, p2/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    lsr z20.b, z12.b, #4
    and z12.b, z12.b, #0x0F
    zip1 z4.b, z12.b, z20.b
    zip2 z5.b, z12.b, z20.b
    sub z4.b, z4.b, #8
    sub z5.b, z5.b, #8
    lsr z21.b, z13.b, #4
    and z13.b, z13.b, #0x0F
    zip1 z6.b, z13.b, z21.b
    zip2 z7.b, z13.b, z21.b
    sub z6.b, z6.b, #8
    sub z7.b, z7.b, #8
    lsr z22.b, z14.b, #4
    and z14.b, z14.b, #0x0F
    zip1 z8.b, z14.b, z22.b
    zip2 z9.b, z14.b, z22.b
    sub z8.b, z8.b, #8
    sub z9.b, z9.b, #8
    lsr z23.b, z15.b, #4
    and z15.b, z15.b, #0x0F
    zip1 z10.b, z15.b, z23.b
    zip2 z11.b, z15.b, z23.b
    sub z10.b, z10.b, #8
    sub z11.b, z11.b, #8
    sdot z16.s, z4.b, z0.b[0]
    sdot z17.s, z5.b, z0.b[0]
    sdot z18.s, z6.b, z0.b[0]
    sdot z19.s, z7.b, z0.b[0]
    sdot z16.s, z8.b, z0.b[1]
    sdot z17.s, z9.b, z0.b[1]
    sdot z18.s, z10.b, z0.b[1]
    sdot z19.s, z11.b, z0.b[1]
    ld1h z12.h, p2/Z, [x23]
    ld1h z13.h, p2/Z, [x23, #1, MUL VL]
    ld1h z14.h, p2/Z, [x23, #2, MUL VL]
    ld1h z15.h, p2/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    lsr z20.b, z12.b, #4
    and z12.b, z12.b, #0x0F
    zip1 z4.b, z12.b, z20.b
    zip2 z5.b, z12.b, z20.b
    sub z4.b, z4.b, #8
    sub z5.b, z5.b, #8
    lsr z21.b, z13.b, #4
    and z13.b, z13.b, #0x0F
    zip1 z6.b, z13.b, z21.b
    zip2 z7.b, z13.b, z21.b
    sub z6.b, z6.b, #8
    sub z7.b, z7.b, #8
    lsr z22.b, z14.b, #4
    and z14.b, z14.b, #0x0F
    zip1 z8.b, z14.b, z22.b
    zip2 z9.b, z14.b, z22.b
    sub z8.b, z8.b, #8
    sub z9.b, z9.b, #8
    lsr z23.b, z15.b, #4
    and z15.b, z15.b, #0x0F
    zip1 z10.b, z15.b, z23.b
    zip2 z11.b, z15.b, z23.b
    sub z10.b, z10.b, #8
    sub z11.b, z11.b, #8
    sdot z16.s, z4.b, z0.b[2]
    sdot z17.s, z5.b, z0.b[2]
    sdot z18.s, z6.b, z0.b[2]
    sdot z19.s, z7.b, z0.b[2]
    sdot z16.s, z8.b, z0.b[3]
    sdot z17.s, z9.b, z0.b[3]
    sdot z18.s, z10.b, z0.b[3]
    sdot z19.s, z11.b, z0.b[3]
    subs x13, x13, #16
    b.gt label_3
    ld1rw z1.s, p2/z, [x21]       // sum
    ld1rw z2.s, p2/z, [x21, #0x4] // scale
    add x21, x21, #8
    ld1w z4.s, p2/Z, [x23]
    ld1w z5.s, p2/Z, [x23, #1, MUL VL]
    ld1w z6.s, p2/Z, [x23, #2, MUL VL]
    ld1w z7.s, p2/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    ld1w z8.s, p2/Z, [x23]
    ld1w z9.s, p2/Z, [x23, #1, MUL VL]
    ld1w z10.s, p2/Z, [x23, #2, MUL VL]
    ld1w z11.s, p2/Z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    fmla z24.s, p2/m, z4.s, z1.s
    fmla z25.s, p2/m, z5.s, z1.s
    fmla z26.s, p2/m, z6.s, z1.s
    fmla z27.s, p2/m, z7.s, z1.s
    fmul z8.s, z8.s, z2.s
    fmul z9.s, z9.s, z2.s
    fmul z10.s, z10.s, z2.s
    fmul z11.s, z11.s, z2.s
    scvtf z16.s, p2/M, z16.s
    scvtf z17.s, p2/M, z17.s
    scvtf z18.s, p2/M, z18.s
    scvtf z19.s, p2/M, z19.s
    fmla z24.s, p2/m, z16.s, z8.s
    fmla z25.s, p2/m, z17.s, z9.s
    fmla z26.s, p2/m, z18.s, z10.s
    fmla z27.s, p2/m, z19.s, z11.s
    add x6, x6, x20
    whilelt p1.s, x6, x19
    b.mi label_2                // b.first label_2
KAI_ASM_LABEL(label_4)
    ld1w z20.s, p2/Z, [x23]
    ld1w z21.s, p2/Z, [x23, #1, MUL VL]
    ld1w z22.s, p2/Z, [x23, #2, MUL VL]
    ld1w z23.s, p2/Z, [x23, #3, MUL VL]
    fadd z24.s, p2/m, z24.s, z20.s
    fadd z25.s, p2/m, z25.s, z21.s
    fadd z26.s, p2/m, z26.s, z22.s
    fadd z27.s, p2/m, z27.s, z23.s
    FMIN z24.s, p2/M, z24.s, z31.s
    FMIN z25.s, p2/M, z25.s, z31.s
    FMIN z26.s, p2/M, z26.s, z31.s
    FMIN z27.s, p2/M, z27.s, z31.s
    FMAX z24.s, p2/M, z24.s, z30.s
    FMAX z25.s, p2/M, z25.s, z30.s
    FMAX z26.s, p2/M, z26.s, z30.s
    FMAX z27.s, p2/M, z27.s, z30.s
    fcvt  z24.h, p2/M, z24.s
    fcvt  z28.h, p2/M, z25.s
    uzp1  z28.h, z24.h, z28.h
    fcvt  z26.h, p2/M, z26.s
    fcvt  z29.h, p2/M, z27.s
    uzp1  z29.h, z26.h, z29.h  
    inch x4, all
    whilelt p5.h, x4, x24
    dech x4, all
    st1h z28.h, p4, [x5]
    st1h z29.h, p5, [x5, #1, MUL VL]
    incb  x4, all
    addvl x5, x5, #2
    ldr x22, [X0, #0x18]        // rhs_stride
    add x10, x10, x22
    whilelt p4.h, x4, x24
    b.mi label_1                // b.first  label_1
KAI_ASM_LABEL(label_5)
    KAI_ASM_INST(0xd503467f)    // smstop
    ldp     d14, d15, [sp, 96]
    ldp     d12, d13, [sp, 80]
    ldp     d10, d11, [sp, 64]
    ldp     d8, d9,   [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp],112
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme1_dot)

    KAI_ASM_END
