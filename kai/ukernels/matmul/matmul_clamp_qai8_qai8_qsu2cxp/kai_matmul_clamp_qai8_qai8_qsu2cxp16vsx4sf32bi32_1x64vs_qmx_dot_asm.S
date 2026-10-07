//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//

#if defined(_MSC_VER) && !defined(__clang__)
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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_qmx_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_qmx_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_qmx_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_qmx_dot)
    // QMX (SME1) GEMV over one packed UINT2 RHS panel per N iteration. The packed layout is:
    //   [ nr*4 corrected bias | roundup(K,32)*nr/4 payload | nr*4 scale ], nr = VL bytes.
    // Each payload byte contains four consecutive K values. SME1 has no ZT0/LUTI2, so shifts,
    // masks and ZIPs expand each byte to signed {-2, -1, 0, 1} SDOT operands.
    //
    // d8-d15 must be preserved: smstart changes the full Z-register state while their low halves are
    // callee-saved by AAPCS64.
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    stp d8, d9, [sp, 48]
    stp d10, d11, [sp, 64]
    stp d12, d13, [sp, 80]
    stp d14, d15, [sp, 96]
    KAI_ASM_INST(0xd503477f)                // smstart
    ptrue p2.b

    ldr x14, [x0, #0x10]                   // lhs
    ldr x13, [x0, #0x18]                   // first packed RHS panel
    ldr x17, [x0, #0x20]                   // N
    ldr x16, [x0, #0x28]                   // K
    ldr x12, [x0, #0x30]                   // dst
    mov x11, #0                            // columns emitted
    cntb x22                               // VL bytes == nr columns

    // x20 is the scale offset and x21 is the complete packed-panel stride.
    add x23, x16, #31
    bic x23, x23, #31                      // roundup(K, 32)
    lsr x23, x23, #2                       // payload size in vectors
    add x23, x23, #4                       // skip four bias vectors
    mul x20, x23, x22
    add x23, x23, #4                       // include four scale vectors
    mul x21, x23, x22

KAI_ASM_LABEL(label_1)  // N loop: one nr-column panel
    addvl x15, x13, #4                     // payload starts after four bias vectors
    ld1w { z4.s }, p2/z, [x13]
    ld1w { z5.s }, p2/z, [x13, #1, MUL VL]
    ld1w { z6.s }, p2/z, [x13, #2, MUL VL]
    ld1w { z7.s }, p2/z, [x13, #3, MUL VL]
    mov x23, #0                            // K offset

KAI_ASM_LABEL(label_2)  // K loop: 16 logical K values, four packed vectors
    whilelt p0.b, x23, x16
    ld1rqb { z0.b }, p0/z, [x14, x23]
    ld1b { z16.b }, p2/z, [x15]
    ld1b { z17.b }, p2/z, [x15, #1, MUL VL]
    ld1b { z18.b }, p2/z, [x15, #2, MUL VL]
    ld1b { z19.b }, p2/z, [x15, #3, MUL VL]

    // Expand K[0..3] from z16.
    orr z24.b, z16.b, z16.b
    lsr z25.b, z16.b, #2
    lsr z26.b, z16.b, #4
    lsr z27.b, z16.b, #6
    and z24.b, z24.b, #0x03
    and z25.b, z25.b, #0x03
    and z26.b, z26.b, #0x03
    and z27.b, z27.b, #0x03
    zip1 z2.b, z24.b, z25.b
    zip2 z3.b, z24.b, z25.b
    zip1 z28.b, z26.b, z27.b
    zip2 z29.b, z26.b, z27.b
    zip1 z24.h, z2.h, z28.h
    zip2 z25.h, z2.h, z28.h
    zip1 z26.h, z3.h, z29.h
    zip2 z27.h, z3.h, z29.h
    sub z24.b, z24.b, #2
    sub z25.b, z25.b, #2
    sub z26.b, z26.b, #2
    sub z27.b, z27.b, #2

    // Expand K[4..7] from z17.
    orr z20.b, z17.b, z17.b
    lsr z21.b, z17.b, #2
    lsr z22.b, z17.b, #4
    lsr z23.b, z17.b, #6
    and z20.b, z20.b, #0x03
    and z21.b, z21.b, #0x03
    and z22.b, z22.b, #0x03
    and z23.b, z23.b, #0x03
    zip1 z2.b, z20.b, z21.b
    zip2 z3.b, z20.b, z21.b
    zip1 z28.b, z22.b, z23.b
    zip2 z29.b, z22.b, z23.b
    zip1 z20.h, z2.h, z28.h
    zip2 z21.h, z2.h, z28.h
    zip1 z22.h, z3.h, z29.h
    zip2 z23.h, z3.h, z29.h
    sub z20.b, z20.b, #2
    sub z21.b, z21.b, #2
    sub z22.b, z22.b, #2
    sub z23.b, z23.b, #2

    sdot z4.s, z24.b, z0.b[0]
    sdot z5.s, z25.b, z0.b[0]
    sdot z6.s, z26.b, z0.b[0]
    sdot z7.s, z27.b, z0.b[0]
    sdot z4.s, z20.b, z0.b[1]
    sdot z5.s, z21.b, z0.b[1]
    sdot z6.s, z22.b, z0.b[1]
    sdot z7.s, z23.b, z0.b[1]

    // Expand K[8..11] from z18.
    orr z8.b, z18.b, z18.b
    lsr z9.b, z18.b, #2
    lsr z10.b, z18.b, #4
    lsr z11.b, z18.b, #6
    and z8.b, z8.b, #0x03
    and z9.b, z9.b, #0x03
    and z10.b, z10.b, #0x03
    and z11.b, z11.b, #0x03
    zip1 z2.b, z8.b, z9.b
    zip2 z3.b, z8.b, z9.b
    zip1 z28.b, z10.b, z11.b
    zip2 z29.b, z10.b, z11.b
    zip1 z8.h, z2.h, z28.h
    zip2 z9.h, z2.h, z28.h
    zip1 z10.h, z3.h, z29.h
    zip2 z11.h, z3.h, z29.h
    sub z8.b, z8.b, #2
    sub z9.b, z9.b, #2
    sub z10.b, z10.b, #2
    sub z11.b, z11.b, #2

    // Expand K[12..15] from z19.
    orr z12.b, z19.b, z19.b
    lsr z13.b, z19.b, #2
    lsr z14.b, z19.b, #4
    lsr z15.b, z19.b, #6
    and z12.b, z12.b, #0x03
    and z13.b, z13.b, #0x03
    and z14.b, z14.b, #0x03
    and z15.b, z15.b, #0x03
    zip1 z2.b, z12.b, z13.b
    zip2 z3.b, z12.b, z13.b
    zip1 z28.b, z14.b, z15.b
    zip2 z29.b, z14.b, z15.b
    zip1 z12.h, z2.h, z28.h
    zip2 z13.h, z2.h, z28.h
    zip1 z14.h, z3.h, z29.h
    zip2 z15.h, z3.h, z29.h
    sub z12.b, z12.b, #2
    sub z13.b, z13.b, #2
    sub z14.b, z14.b, #2
    sub z15.b, z15.b, #2

    sdot z4.s, z8.b, z0.b[2]
    sdot z5.s, z9.b, z0.b[2]
    sdot z6.s, z10.b, z0.b[2]
    sdot z7.s, z11.b, z0.b[2]
    sdot z4.s, z12.b, z0.b[3]
    sdot z5.s, z13.b, z0.b[3]
    sdot z6.s, z14.b, z0.b[3]
    sdot z7.s, z15.b, z0.b[3]

    addvl x15, x15, #4
    add x23, x23, #16
    cmp x23, x16
    blt label_2

    // Requantize four accumulator vectors and narrow them to one nr-byte result vector.
    add x24, x13, x20
    ld1w { z24.s }, p2/z, [x24]
    ld1w { z25.s }, p2/z, [x24, #1, MUL VL]
    ld1w { z26.s }, p2/z, [x24, #2, MUL VL]
    ld1w { z27.s }, p2/z, [x24, #3, MUL VL]
    ld1rw { z28.s }, p2/z, [x0]            // destination zero point
    ld1rw { z29.s }, p2/z, [x0, #8]        // clamp min
    ld1rw { z30.s }, p2/z, [x0, #4]        // clamp max
    scvtf z4.s, p2/m, z4.s
    scvtf z5.s, p2/m, z5.s
    scvtf z6.s, p2/m, z6.s
    scvtf z7.s, p2/m, z7.s
    fmul z4.s, z4.s, z24.s
    fmul z5.s, z5.s, z25.s
    fmul z6.s, z6.s, z26.s
    fmul z7.s, z7.s, z27.s
    frintn z4.s, p2/m, z4.s
    frintn z5.s, p2/m, z5.s
    frintn z6.s, p2/m, z6.s
    frintn z7.s, p2/m, z7.s
    fcvtzs z4.s, p2/m, z4.s
    fcvtzs z5.s, p2/m, z5.s
    fcvtzs z6.s, p2/m, z6.s
    fcvtzs z7.s, p2/m, z7.s
    add z4.s, z4.s, z28.s
    add z5.s, z5.s, z28.s
    add z6.s, z6.s, z28.s
    add z7.s, z7.s, z28.s
    sclamp z4.s, z29.s, z30.s
    sclamp z5.s, z29.s, z30.s
    sclamp z6.s, z29.s, z30.s
    sclamp z7.s, z29.s, z30.s
    uzp1 z4.h, z4.h, z5.h
    uzp1 z6.h, z6.h, z7.h
    uzp1 z4.b, z4.b, z6.b
    whilelt p1.b, x11, x17
    st1b { z4.b }, p1, [x12]

    add x13, x13, x21
    add x12, x12, x22
    incb x11
    cmp x11, x17
    blt label_1

    KAI_ASM_INST(0xd503467f)                // smstop
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp d8, d9, [sp, 48]
    ldp d10, d11, [sp, 64]
    ldp d12, d13, [sp, 80]
    ldp d14, d15, [sp, 96]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8_qsu2cxp16vsx4sf32bi32_1x64vs_qmx_dot)

    KAI_ASM_END
