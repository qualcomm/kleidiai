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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8p8vsx4_qsi2cxp16vsx4s12s8s4s0sf32bi32_8vsx16vs_qmx_mopa)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsi2cxp16vsx4s12s8s4s0sf32bi32_8vsx16vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsi2cxp16vsx4s12s8s4s0sf32bi32_8vsx16vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsi2cxp16vsx4s12s8s4s0sf32bi32_8vsx16vs_qmx_mopa)
    stp x19, x20, [sp, -128]!
    stp x21, x22, [sp, 16]
    stp x23, x24, [sp, 32]
    str x28, [sp, 48]
    stp d8, d9, [sp, 64]
    stp d10, d11, [sp, 80]
    stp d12, d13, [sp, 96]
    stp d14, d15, [sp, 112]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ptrue p1.b
    cntw x6
    lsl x7, x6, #1
    lsl x5, x6, #2
    ldr x19, [x0, #0x0]
    ldr x20, [x0, #0x8]
    ldr x21, [x0, #0x10]
    ldr x24, [x0, #0x18]
    ldr x12, [x0, #0x20]
    ldr x13, [x0, #0x28]
    ldr x4, [x0, #0x30]
    ldr x22, [x0, #0x48]
    ldr x23, [x0, #0x50]
    KAI_ASM_INST(0x854ec419)  // ld1rw { z25.s }, p1/Z, [x0, #56]
    KAI_ASM_INST(0x854fc41a)  // ld1rw { z26.s }, p1/Z, [x0, #60]
    KAI_ASM_INST(0x8550c41b)  // ld1rw { z27.s }, p1/Z, [x0, #64]
KAI_ASM_LABEL(label_1)  // M loop
    mov x8, x20
    mov x9, x21
    mov x28, x13
KAI_ASM_LABEL(label_2)  // N loop
    mov x1, #0
    whilelt p4.s, x1, x28
    mov x1, x6
    whilelt p5.s, x1, x28
    lsl x1, x6, #1
    whilelt p6.s, x1, x28
    add x1, x1, x6
    whilelt p7.s, x1, x28
    whilelt p3.b, xzr, x28
    ld1w { z20.s }, p4/Z, [x8]
    ld1w { z21.s }, p5/Z, [x8, #1, MUL VL]
    ld1w { z22.s }, p6/Z, [x8, #2, MUL VL]
    ld1w { z23.s }, p7/Z, [x8, #3, MUL VL]
    mov x17, #0
KAI_ASM_LABEL(label_3)  // M tile loop
    mul x16, x17, x6
    whilelt p0.s, x16, x12
    b.eq label_6
    KAI_ASM_INST(0xc00800ff)  // zero {za}
    KAI_ASM_INST(0xc0902680)  // addha za0.s, p1/M, p1/M, z20.s
    KAI_ASM_INST(0xc09026a1)  // addha za1.s, p1/M, p1/M, z21.s
    KAI_ASM_INST(0xc09026c2)  // addha za2.s, p1/M, p1/M, z22.s
    KAI_ASM_INST(0xc09026e3)  // addha za3.s, p1/M, p1/M, z23.s
    lsl x2, x6, #2
    madd x10, x17, x2, x19
    mov x11, x8
    addvl x11, x11, #4
    add x14, x4, #31
    bic x14, x14, #31
    lsr x14, x14, #2
    madd x3, x14, x5, x11
    add x14, x4, #3
    lsr x14, x14, #2
    lsr x18, x14, #2
    and x14, x14, #3
    cbz x18, label_7
KAI_ASM_LABEL(label_4)  // K loop: one signed s12s8s4s0 16-K unit
    ld1w { z4.s }, p0/Z, [x10]
    add x10, x10, x7, LSL #2
    ld1w { z5.s }, p0/Z, [x10]
    add x10, x10, x7, LSL #2
    ld1w { z6.s }, p0/Z, [x10]
    add x10, x10, x7, LSL #2
    ld1w { z7.s }, p0/Z, [x10]
    add x10, x10, x7, LSL #2
    ld1b { z8.b }, p1/Z, [x11]
    ld1b { z9.b }, p1/Z, [x11, #1, MUL VL]
    ld1b { z10.b }, p1/Z, [x11, #2, MUL VL]
    ld1b { z11.b }, p1/Z, [x11, #3, MUL VL]
    addvl x11, x11, #4

    // Low fields contain K+0..3.
    lsl z12.b, z8.b, #6
    asr z12.b, z12.b, #6
    lsl z13.b, z9.b, #6
    asr z13.b, z13.b, #6
    lsl z14.b, z10.b, #6
    asr z14.b, z14.b, #6
    lsl z15.b, z11.b, #6
    asr z15.b, z15.b, #6
    KAI_ASM_INST(0xa08c2480)  // smopa za0.s, p1/M, p1/M, z4.b, z12.b
    KAI_ASM_INST(0xa08d2481)  // smopa za1.s, p1/M, p1/M, z4.b, z13.b
    KAI_ASM_INST(0xa08e2482)  // smopa za2.s, p1/M, p1/M, z4.b, z14.b
    KAI_ASM_INST(0xa08f2483)  // smopa za3.s, p1/M, p1/M, z4.b, z15.b

    // The remaining fields contain K+4..7, K+8..11, and K+12..15.
    lsl z12.b, z8.b, #4
    asr z12.b, z12.b, #6
    lsl z13.b, z9.b, #4
    asr z13.b, z13.b, #6
    lsl z14.b, z10.b, #4
    asr z14.b, z14.b, #6
    lsl z15.b, z11.b, #4
    asr z15.b, z15.b, #6
    KAI_ASM_INST(0xa08c24a0)  // smopa za0.s, p1/M, p1/M, z5.b, z12.b
    KAI_ASM_INST(0xa08d24a1)  // smopa za1.s, p1/M, p1/M, z5.b, z13.b
    KAI_ASM_INST(0xa08e24a2)  // smopa za2.s, p1/M, p1/M, z5.b, z14.b
    KAI_ASM_INST(0xa08f24a3)  // smopa za3.s, p1/M, p1/M, z5.b, z15.b
    lsl z12.b, z8.b, #2
    asr z12.b, z12.b, #6
    lsl z13.b, z9.b, #2
    asr z13.b, z13.b, #6
    lsl z14.b, z10.b, #2
    asr z14.b, z14.b, #6
    lsl z15.b, z11.b, #2
    asr z15.b, z15.b, #6
    KAI_ASM_INST(0xa08c24c0)  // smopa za0.s, p1/M, p1/M, z6.b, z12.b
    KAI_ASM_INST(0xa08d24c1)  // smopa za1.s, p1/M, p1/M, z6.b, z13.b
    KAI_ASM_INST(0xa08e24c2)  // smopa za2.s, p1/M, p1/M, z6.b, z14.b
    KAI_ASM_INST(0xa08f24c3)  // smopa za3.s, p1/M, p1/M, z6.b, z15.b
    asr z12.b, z8.b, #6
    asr z13.b, z9.b, #6
    asr z14.b, z10.b, #6
    asr z15.b, z11.b, #6
    KAI_ASM_INST(0xa08c24e0)  // smopa za0.s, p1/M, p1/M, z7.b, z12.b
    KAI_ASM_INST(0xa08d24e1)  // smopa za1.s, p1/M, p1/M, z7.b, z13.b
    KAI_ASM_INST(0xa08e24e2)  // smopa za2.s, p1/M, p1/M, z7.b, z14.b
    KAI_ASM_INST(0xa08f24e3)  // smopa za3.s, p1/M, p1/M, z7.b, z15.b
    subs x18, x18, #1
    b.gt label_4
KAI_ASM_LABEL(label_7)  // K tail: 1..3 four-K groups
    cbz x14, label_8
    ld1b { z8.b }, p1/Z, [x11]
    ld1b { z9.b }, p1/Z, [x11, #1, MUL VL]
    ld1b { z10.b }, p1/Z, [x11, #2, MUL VL]
    ld1b { z11.b }, p1/Z, [x11, #3, MUL VL]
    ld1w { z4.s }, p0/Z, [x10]
    add x10, x10, x7, LSL #2
    lsl z12.b, z8.b, #6
    asr z12.b, z12.b, #6
    lsl z13.b, z9.b, #6
    asr z13.b, z13.b, #6
    lsl z14.b, z10.b, #6
    asr z14.b, z14.b, #6
    lsl z15.b, z11.b, #6
    asr z15.b, z15.b, #6
    KAI_ASM_INST(0xa08c2480)  // smopa za0.s, p1/M, p1/M, z4.b, z12.b
    KAI_ASM_INST(0xa08d2481)  // smopa za1.s, p1/M, p1/M, z4.b, z13.b
    KAI_ASM_INST(0xa08e2482)  // smopa za2.s, p1/M, p1/M, z4.b, z14.b
    KAI_ASM_INST(0xa08f2483)  // smopa za3.s, p1/M, p1/M, z4.b, z15.b
    cmp x14, #1
    b.eq label_8
    ld1w { z5.s }, p0/Z, [x10]
    add x10, x10, x7, LSL #2
    lsl z12.b, z8.b, #4
    asr z12.b, z12.b, #6
    lsl z13.b, z9.b, #4
    asr z13.b, z13.b, #6
    lsl z14.b, z10.b, #4
    asr z14.b, z14.b, #6
    lsl z15.b, z11.b, #4
    asr z15.b, z15.b, #6
    KAI_ASM_INST(0xa08c24a0)  // smopa za0.s, p1/M, p1/M, z5.b, z12.b
    KAI_ASM_INST(0xa08d24a1)  // smopa za1.s, p1/M, p1/M, z5.b, z13.b
    KAI_ASM_INST(0xa08e24a2)  // smopa za2.s, p1/M, p1/M, z5.b, z14.b
    KAI_ASM_INST(0xa08f24a3)  // smopa za3.s, p1/M, p1/M, z5.b, z15.b
    cmp x14, #2
    b.eq label_8
    ld1w { z6.s }, p0/Z, [x10]
    lsl z12.b, z8.b, #2
    asr z12.b, z12.b, #6
    lsl z13.b, z9.b, #2
    asr z13.b, z13.b, #6
    lsl z14.b, z10.b, #2
    asr z14.b, z14.b, #6
    lsl z15.b, z11.b, #2
    asr z15.b, z15.b, #6
    KAI_ASM_INST(0xa08c24c0)  // smopa za0.s, p1/M, p1/M, z6.b, z12.b
    KAI_ASM_INST(0xa08d24c1)  // smopa za1.s, p1/M, p1/M, z6.b, z13.b
    KAI_ASM_INST(0xa08e24c2)  // smopa za2.s, p1/M, p1/M, z6.b, z14.b
    KAI_ASM_INST(0xa08f24c3)  // smopa za3.s, p1/M, p1/M, z6.b, z15.b
KAI_ASM_LABEL(label_8)  // K loop end
    ld1w { z4.s }, p4/Z, [x3]
    ld1w { z5.s }, p5/Z, [x3, #1, MUL VL]
    ld1w { z6.s }, p6/Z, [x3, #2, MUL VL]
    ld1w { z7.s }, p7/Z, [x3, #3, MUL VL]
    sub x3, x12, x16
    cmp x3, x6
    csel x3, x3, x6, LT
    madd x15, x16, x24, x9
    mov x14, #0
KAI_ASM_LABEL(label_5)  // Output row loop
    KAI_ASM_INST(0xc002440c)  // mova z12.b, p1/M, za0h.b[w14, 0]
    KAI_ASM_INST(0xc002442d)  // mova z13.b, p1/M, za0h.b[w14, 1]
    KAI_ASM_INST(0xc002444e)  // mova z14.b, p1/M, za0h.b[w14, 2]
    KAI_ASM_INST(0xc002446f)  // mova z15.b, p1/M, za0h.b[w14, 3]
    scvtf z12.s, p1/M, z12.s
    scvtf z13.s, p1/M, z13.s
    scvtf z14.s, p1/M, z14.s
    scvtf z15.s, p1/M, z15.s
    fmul z12.s, z12.s, z4.s
    fmul z13.s, z13.s, z5.s
    fmul z14.s, z14.s, z6.s
    fmul z15.s, z15.s, z7.s
    frintn z12.s, p1/M, z12.s
    frintn z13.s, p1/M, z13.s
    frintn z14.s, p1/M, z14.s
    frintn z15.s, p1/M, z15.s
    fcvtzs z12.s, p1/M, z12.s
    fcvtzs z13.s, p1/M, z13.s
    fcvtzs z14.s, p1/M, z14.s
    fcvtzs z15.s, p1/M, z15.s
    add z12.s, p1/M, z12.s, z27.s
    add z13.s, p1/M, z13.s, z27.s
    add z14.s, p1/M, z14.s, z27.s
    add z15.s, p1/M, z15.s, z27.s
    KAI_ASM_INST(0x449ac32c)  // sclamp z12.s, z25.s, z26.s
    KAI_ASM_INST(0x449ac32d)  // sclamp z13.s, z25.s, z26.s
    KAI_ASM_INST(0x449ac32e)  // sclamp z14.s, z25.s, z26.s
    KAI_ASM_INST(0x449ac32f)  // sclamp z15.s, z25.s, z26.s
    st1b { z12.s }, p4, [x15]
    st1b { z13.s }, p5, [x15, x6]
    add x1, x15, x6, LSL #1
    st1b { z14.s }, p6, [x1]
    add x1, x1, x6
    st1b { z15.s }, p7, [x1]
    add x15, x15, x24
    add x14, x14, #4
    cmp x14, x3, LSL #2
    b.lt label_5
KAI_ASM_LABEL(label_6)  // M tile loop end
    add x17, x17, #1
    cmp x17, #2
    b.lt label_3
    add x8, x8, x23
    add x9, x9, x5
    subs x28, x28, x5
    b.gt label_2
    add x19, x19, x22
    mul x1, x7, x24
    add x21, x21, x1
    subs x12, x12, x7
    b.gt label_1
    KAI_ASM_INST(0xd503467f)  // SMSTOP
    ldp d14, d15, [sp, 112]
    ldp d12, d13, [sp, 96]
    ldp d10, d11, [sp, 80]
    ldp d8, d9, [sp, 64]
    ldr x28, [sp, 48]
    ldp x23, x24, [sp, 32]
    ldp x21, x22, [sp, 16]
    ldp x19, x20, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8p8vsx4_qsi2cxp16vsx4s12s8s4s0sf32bi32_8vsx16vs_qmx_mopa)

    KAI_ASM_END
