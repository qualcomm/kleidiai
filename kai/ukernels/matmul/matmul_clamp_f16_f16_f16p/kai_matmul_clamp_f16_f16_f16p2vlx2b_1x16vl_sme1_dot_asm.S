//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

    KAI_ASM_CODE(matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)

    KAI_ASM_GLOBAL(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)

KAI_ASM_FUNCTION_TYPE(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)
KAI_ASM_FUNCTION_LABEL(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)
    fcvt h0, s0
    fmov w0, h0
    ret
    KAI_ASM_FUNCTION_END(kai_f16_from_float_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)


KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    SMSTART 
    mov x8, #0x0
    ldr x5, [x0, #0x18]
    cntw x6, ALL, MUL #4
    ptrue p1.b
    ldr x7, [x0, #0x20]
    mov x22, #0x1
    ldr x21, [x0, #0x10]
    add x17, x5, x6
    ldr x20, [x0, #0x28]
    sub x17, x17, #0x1
    ldr x16, [x0, #0x8]
    udiv x17, x17, x6
    ldr x15, [x0, #0x30]
    mov x14, x21
    add x21, x17, #0x3
    mov x13, x20
    and x21, x21, #0xfffffffffffffffc
    mul x21, x21, x6
    mul x21, x21, x7
    lsl x21, x21, #0x1
KAI_ASM_LABEL(label_1)  // RHS size check loop
    cmp x21, #0x200, LSL #12
    blt label_2
    tbnz x21, #0, label_3
    lsr x21, x21, #0x1
    lsl x22, x22, #0x1
    b label_1
KAI_ASM_LABEL(label_2)  // RHS do prefetch
    lsl x20, x21, #0x26
    sub x22, x22, #0x1
    lsl x22, x22, #0x16
    orr x21, x21, x20
    orr x21, x21, x22
    rprfm #2, x21, [x14]
KAI_ASM_LABEL(label_3)  // RHS prefetch exit
    add x12, x7, #0x1
    cntw x20, ALL, MUL #2
    bic x12, x12, #0x1
    lsl x12, x12, #0x1
    add x12, x12, #0x2
    mul x12, x12, x20
KAI_ASM_LABEL(label_4)  // Column loop
    cmp x17, #0x4
    bge label_22
    cmp x17, #0x2
    bgt label_16
    beq label_10
    cntw x20, ALL, MUL #2
    add x22, x14, x12
    ld1h z4.s, p1/Z, [x14]
    ld1h z5.s, p1/Z, [x14, #1, MUL VL]
    cmp x5, x20
    mov x11, x7
    csel x22, x22, x14, GT
    mov x21, x5
    fcvt z4.s, p1/M, z4.h
    fcvt z5.s, p1/M, z5.h
    fcvt z6.s, p1/M, z6.h
    fcvt z7.s, p1/M, z7.h
    mov x10, x16
    lsl x20, x7, #0x1
    mov x4,XZR
    whilelt p2.h, x4, x21
    inch x4
    whilelt p3.h, x4, x21
    cmp x11, #0x8
    rprfm pldkeep, x20, [x10]
    inch x14, ALL, MUL #2
    inch x22, ALL, MUL #2
    ble label_6
KAI_ASM_LABEL(label_5)  // Width 1: Multiply loop: Main loop head
    whilelt p0.h, XZR, x11
    ld1rqh { z3.h }, p0/Z, [x10]
    sub x11, x11, #0x8
    add x10, x10, #0x10
    cmp x11, #0x8
    ldnt1h z12.h, p1/Z, [x14]
    ldnt1h z13.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z14.h, p1/Z, [x22]
    ldnt1h z15.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmlalb z4.s, z12.h, z3.h[0]
    fmlalb z5.s, z13.h, z3.h[0]
    fmlalb z6.s, z14.h, z3.h[0]
    fmlalb z7.s, z15.h, z3.h[0]
    fmlalt z4.s, z12.h, z3.h[1]
    fmlalt z5.s, z13.h, z3.h[1]
    fmlalt z6.s, z14.h, z3.h[1]
    fmlalt z7.s, z15.h, z3.h[1]
    ldnt1h z12.h, p1/Z, [x14]
    ldnt1h z13.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z14.h, p1/Z, [x22]
    ldnt1h z15.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmlalb z4.s, z12.h, z3.h[2]
    fmlalb z5.s, z13.h, z3.h[2]
    fmlalb z6.s, z14.h, z3.h[2]
    fmlalb z7.s, z15.h, z3.h[2]
    fmlalt z4.s, z12.h, z3.h[3]
    fmlalt z5.s, z13.h, z3.h[3]
    fmlalt z6.s, z14.h, z3.h[3]
    fmlalt z7.s, z15.h, z3.h[3]
    ldnt1h z12.h, p1/Z, [x14]
    ldnt1h z13.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z14.h, p1/Z, [x22]
    ldnt1h z15.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmlalb z4.s, z12.h, z3.h[4]
    fmlalb z5.s, z13.h, z3.h[4]
    fmlalb z6.s, z14.h, z3.h[4]
    fmlalb z7.s, z15.h, z3.h[4]
    fmlalt z4.s, z12.h, z3.h[5]
    fmlalt z5.s, z13.h, z3.h[5]
    fmlalt z6.s, z14.h, z3.h[5]
    fmlalt z7.s, z15.h, z3.h[5]
    ldnt1h z12.h, p1/Z, [x14]
    ldnt1h z13.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z14.h, p1/Z, [x22]
    ldnt1h z15.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmlalb z4.s, z12.h, z3.h[6]
    fmlalb z5.s, z13.h, z3.h[6]
    fmlalb z6.s, z14.h, z3.h[6]
    fmlalb z7.s, z15.h, z3.h[6]
    fmlalt z4.s, z12.h, z3.h[7]
    fmlalt z5.s, z13.h, z3.h[7]
    fmlalt z6.s, z14.h, z3.h[7]
    fmlalt z7.s, z15.h, z3.h[7]
    bgt label_5
KAI_ASM_LABEL(label_6)  // Width 1: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x11
    subs x11, x11, #0x2
    ld1rqh { z3.h }, p0/Z, [x10]
    ldnt1h z12.h, p1/Z, [x14]
    ldnt1h z13.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z14.h, p1/Z, [x22]
    ldnt1h z15.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmlalb z4.s, z12.h, z3.h[0]
    fmlalb z5.s, z13.h, z3.h[0]
    fmlalb z6.s, z14.h, z3.h[0]
    fmlalb z7.s, z15.h, z3.h[0]
    fmlalt z4.s, z12.h, z3.h[1]
    fmlalt z5.s, z13.h, z3.h[1]
    fmlalt z6.s, z14.h, z3.h[1]
    fmlalt z7.s, z15.h, z3.h[1]
    ble label_7
    subs x11, x11, #0x2
    ldnt1h z12.h, p1/Z, [x14]
    ldnt1h z13.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z14.h, p1/Z, [x22]
    ldnt1h z15.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmlalb z4.s, z12.h, z3.h[2]
    fmlalb z5.s, z13.h, z3.h[2]
    fmlalb z6.s, z14.h, z3.h[2]
    fmlalb z7.s, z15.h, z3.h[2]
    fmlalt z4.s, z12.h, z3.h[3]
    fmlalt z5.s, z13.h, z3.h[3]
    fmlalt z6.s, z14.h, z3.h[3]
    fmlalt z7.s, z15.h, z3.h[3]
    ble label_7
    subs x11, x11, #0x2
    ldnt1h z12.h, p1/Z, [x14]
    ldnt1h z13.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z14.h, p1/Z, [x22]
    ldnt1h z15.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmlalb z4.s, z12.h, z3.h[4]
    fmlalb z5.s, z13.h, z3.h[4]
    fmlalb z6.s, z14.h, z3.h[4]
    fmlalb z7.s, z15.h, z3.h[4]
    fmlalt z4.s, z12.h, z3.h[5]
    fmlalt z5.s, z13.h, z3.h[5]
    fmlalt z6.s, z14.h, z3.h[5]
    fmlalt z7.s, z15.h, z3.h[5]
    ble label_7
    ldnt1h z12.h, p1/Z, [x14]
    ldnt1h z13.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z14.h, p1/Z, [x22]
    ldnt1h z15.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    fmlalb z4.s, z12.h, z3.h[6]
    fmlalb z5.s, z13.h, z3.h[6]
    fmlalb z6.s, z14.h, z3.h[6]
    fmlalb z7.s, z15.h, z3.h[6]
    fmlalt z4.s, z12.h, z3.h[7]
    fmlalt z5.s, z13.h, z3.h[7]
    fmlalt z6.s, z14.h, z3.h[7]
    fmlalt z7.s, z15.h, z3.h[7]
KAI_ASM_LABEL(label_7)  // Width 1: Multiply loop: multiply skip
    tbz x15, #1, label_8
    add x21, x0, #0x2
    add x20, x0, #0x0
    ld1rh { z19.h }, p1/Z, [x21]
    ld1rh { z22.h }, p1/Z, [x20]
    fcvt z4.h, p1/M, z4.s
    fcvt z20.h, p1/M, z5.s
    uzp1 z20.h, z4.h, z20.h
    fcvt z6.h, p1/M, z6.s
    fcvt z21.h, p1/M, z7.s
    uzp1 z21.h, z6.h, z21.h
    FMIN z20.h, p1/M, z20.h, z22.h
    FMAX z20.h, p1/M, z20.h, z19.h
    FMIN z21.h, p1/M, z21.h, z22.h
    FMAX z21.h, p1/M, z21.h, z19.h

    st1h z20.h, p2, [x13]
    st1h z21.h, p3, [x13, #1, MUL VL]
    b label_9
KAI_ASM_LABEL(label_8)  // Width 1: No activation
    fcvt z5.h, p1/M, z4.s
    fcvt z4.h, p1/M, z5.s
    uzp1 z4.h, z5.h, z4.h
    fcvt z6.h, p1/M, z6.s
    fcvt z5.h, p1/M, z7.s
    uzp1 z5.h, z6.h, z5.h
    st1h z4.h, p2, [x13]
    st1h z5.h, p3, [x13, #1, MUL VL]
KAI_ASM_LABEL(label_9)  // Width 1: Output done
    b label_28
KAI_ASM_LABEL(label_10)  // Width 2
    add x24, x14, x12, LSL #1
    cntw x20, ALL, MUL #6
    ld1h z8.s, p1/Z, [x14]
    ld1h z9.s, p1/Z, [x14, #1, MUL VL]
    add x23, x24, x12
    cmp x5, x20
    add x22, x14, x12
    csel x23, x23, x14, GT
    ld1h z12.s, p1/Z, [x24]
    ld1h z13.s, p1/Z, [x24, #1, MUL VL]
    ld1h z10.s, p1/Z, [x22]
    ld1h z11.s, p1/Z, [x22, #1, MUL VL]
    ld1h z14.s, p1/Z, [x23]
    ld1h z15.s, p1/Z, [x23, #1, MUL VL]
    fcvt z8.s, p1/M, z8.h
    fcvt z9.s, p1/M, z9.h
    fcvt z10.s, p1/M, z10.h
    fcvt z11.s, p1/M, z11.h
    fcvt z12.s, p1/M, z12.h
    fcvt z13.s, p1/M, z13.h
    fcvt z14.s, p1/M, z14.h
    fcvt z15.s, p1/M, z15.h
    mov x11, x7
    sub x21, x5, x6
    mov x10, x16
    lsl x20, x7, #0x1
    mov x4,XZR
    whilelt p2.h, x4, x21
    inch x4
    whilelt p3.h, x4, x21
    cmp x11, #0x8
    rprfm pldkeep, x20, [x10]
    inch x14, ALL, MUL #2
    inch x22, ALL, MUL #2
    inch x24, ALL, MUL #2
    inch x23, ALL, MUL #2
    ble label_12
KAI_ASM_LABEL(label_11)  // Width 2: Multiply loop: Main loop head
    whilelt p0.h, XZR, x11
    ld1rqh { z3.h }, p0/Z, [x10]
    sub x11, x11, #0x8
    add x10, x10, #0x10
    cmp x11, #0x8
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x22]
    ldnt1h z19.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z20.h, p1/Z, [x24]
    ldnt1h z21.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1h z22.h, p1/Z, [x23]
    ldnt1h z23.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmlalb z8.s, z16.h, z3.h[0]
    fmlalb z9.s, z17.h, z3.h[0]
    fmlalb z10.s, z18.h, z3.h[0]
    fmlalb z11.s, z19.h, z3.h[0]
    fmlalt z8.s, z16.h, z3.h[1]
    fmlalt z9.s, z17.h, z3.h[1]
    fmlalt z10.s, z18.h, z3.h[1]
    fmlalt z11.s, z19.h, z3.h[1]
    fmlalb z12.s, z20.h, z3.h[0]
    fmlalb z13.s, z21.h, z3.h[0]
    fmlalb z14.s, z22.h, z3.h[0]
    fmlalb z15.s, z23.h, z3.h[0]
    fmlalt z12.s, z20.h, z3.h[1]
    fmlalt z13.s, z21.h, z3.h[1]
    fmlalt z14.s, z22.h, z3.h[1]
    fmlalt z15.s, z23.h, z3.h[1]
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x22]
    ldnt1h z19.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z20.h, p1/Z, [x24]
    ldnt1h z21.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1h z22.h, p1/Z, [x23]
    ldnt1h z23.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmlalb z8.s, z16.h, z3.h[2]
    fmlalb z9.s, z17.h, z3.h[2]
    fmlalb z10.s, z18.h, z3.h[2]
    fmlalb z11.s, z19.h, z3.h[2]
    fmlalt z8.s, z16.h, z3.h[3]
    fmlalt z9.s, z17.h, z3.h[3]
    fmlalt z10.s, z18.h, z3.h[3]
    fmlalt z11.s, z19.h, z3.h[3]
    fmlalb z12.s, z20.h, z3.h[2]
    fmlalb z13.s, z21.h, z3.h[2]
    fmlalb z14.s, z22.h, z3.h[2]
    fmlalb z15.s, z23.h, z3.h[2]
    fmlalt z12.s, z20.h, z3.h[3]
    fmlalt z13.s, z21.h, z3.h[3]
    fmlalt z14.s, z22.h, z3.h[3]
    fmlalt z15.s, z23.h, z3.h[3]
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x22]
    ldnt1h z19.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z20.h, p1/Z, [x24]
    ldnt1h z21.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1h z22.h, p1/Z, [x23]
    ldnt1h z23.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmlalb z8.s, z16.h, z3.h[4]
    fmlalb z9.s, z17.h, z3.h[4]
    fmlalb z10.s, z18.h, z3.h[4]
    fmlalb z11.s, z19.h, z3.h[4]
    fmlalt z8.s, z16.h, z3.h[5]
    fmlalt z9.s, z17.h, z3.h[5]
    fmlalt z10.s, z18.h, z3.h[5]
    fmlalt z11.s, z19.h, z3.h[5]
    fmlalb z12.s, z20.h, z3.h[4]
    fmlalb z13.s, z21.h, z3.h[4]
    fmlalb z14.s, z22.h, z3.h[4]
    fmlalb z15.s, z23.h, z3.h[4]
    fmlalt z12.s, z20.h, z3.h[5]
    fmlalt z13.s, z21.h, z3.h[5]
    fmlalt z14.s, z22.h, z3.h[5]
    fmlalt z15.s, z23.h, z3.h[5]
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x22]
    ldnt1h z19.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z20.h, p1/Z, [x24]
    ldnt1h z21.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1h z22.h, p1/Z, [x23]
    ldnt1h z23.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmlalb z8.s, z16.h, z3.h[6]
    fmlalb z9.s, z17.h, z3.h[6]
    fmlalb z10.s, z18.h, z3.h[6]
    fmlalb z11.s, z19.h, z3.h[6]
    fmlalt z8.s, z16.h, z3.h[7]
    fmlalt z9.s, z17.h, z3.h[7]
    fmlalt z10.s, z18.h, z3.h[7]
    fmlalt z11.s, z19.h, z3.h[7]
    fmlalb z12.s, z20.h, z3.h[6]
    fmlalb z13.s, z21.h, z3.h[6]
    fmlalb z14.s, z22.h, z3.h[6]
    fmlalb z15.s, z23.h, z3.h[6]
    fmlalt z12.s, z20.h, z3.h[7]
    fmlalt z13.s, z21.h, z3.h[7]
    fmlalt z14.s, z22.h, z3.h[7]
    fmlalt z15.s, z23.h, z3.h[7]
    bgt label_11
KAI_ASM_LABEL(label_12)  // Width 2: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x11
    subs x11, x11, #0x2
    ld1rqh { z3.h }, p0/Z, [x10]
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x22]
    ldnt1h z19.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z20.h, p1/Z, [x24]
    ldnt1h z21.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1h z22.h, p1/Z, [x23]
    ldnt1h z23.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmlalb z8.s, z16.h, z3.h[0]
    fmlalb z9.s, z17.h, z3.h[0]
    fmlalb z10.s, z18.h, z3.h[0]
    fmlalb z11.s, z19.h, z3.h[0]
    fmlalt z8.s, z16.h, z3.h[1]
    fmlalt z9.s, z17.h, z3.h[1]
    fmlalt z10.s, z18.h, z3.h[1]
    fmlalt z11.s, z19.h, z3.h[1]
    fmlalb z12.s, z20.h, z3.h[0]
    fmlalb z13.s, z21.h, z3.h[0]
    fmlalb z14.s, z22.h, z3.h[0]
    fmlalb z15.s, z23.h, z3.h[0]
    fmlalt z12.s, z20.h, z3.h[1]
    fmlalt z13.s, z21.h, z3.h[1]
    fmlalt z14.s, z22.h, z3.h[1]
    fmlalt z15.s, z23.h, z3.h[1]
    ble label_13
    subs x11, x11, #0x2
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x22]
    ldnt1h z19.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z20.h, p1/Z, [x24]
    ldnt1h z21.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1h z22.h, p1/Z, [x23]
    ldnt1h z23.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmlalb z8.s, z16.h, z3.h[2]
    fmlalb z9.s, z17.h, z3.h[2]
    fmlalb z10.s, z18.h, z3.h[2]
    fmlalb z11.s, z19.h, z3.h[2]
    fmlalt z8.s, z16.h, z3.h[3]
    fmlalt z9.s, z17.h, z3.h[3]
    fmlalt z10.s, z18.h, z3.h[3]
    fmlalt z11.s, z19.h, z3.h[3]
    fmlalb z12.s, z20.h, z3.h[2]
    fmlalb z13.s, z21.h, z3.h[2]
    fmlalb z14.s, z22.h, z3.h[2]
    fmlalb z15.s, z23.h, z3.h[2]
    fmlalt z12.s, z20.h, z3.h[3]
    fmlalt z13.s, z21.h, z3.h[3]
    fmlalt z14.s, z22.h, z3.h[3]
    fmlalt z15.s, z23.h, z3.h[3]
    ble label_13
    subs x11, x11, #0x2
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x22]
    ldnt1h z19.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z20.h, p1/Z, [x24]
    ldnt1h z21.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1h z22.h, p1/Z, [x23]
    ldnt1h z23.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmlalb z8.s, z16.h, z3.h[4]
    fmlalb z9.s, z17.h, z3.h[4]
    fmlalb z10.s, z18.h, z3.h[4]
    fmlalb z11.s, z19.h, z3.h[4]
    fmlalt z8.s, z16.h, z3.h[5]
    fmlalt z9.s, z17.h, z3.h[5]
    fmlalt z10.s, z18.h, z3.h[5]
    fmlalt z11.s, z19.h, z3.h[5]
    fmlalb z12.s, z20.h, z3.h[4]
    fmlalb z13.s, z21.h, z3.h[4]
    fmlalb z14.s, z22.h, z3.h[4]
    fmlalb z15.s, z23.h, z3.h[4]
    fmlalt z12.s, z20.h, z3.h[5]
    fmlalt z13.s, z21.h, z3.h[5]
    fmlalt z14.s, z22.h, z3.h[5]
    fmlalt z15.s, z23.h, z3.h[5]
    ble label_13
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x22]
    ldnt1h z19.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z20.h, p1/Z, [x24]
    ldnt1h z21.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    ldnt1h z22.h, p1/Z, [x23]
    ldnt1h z23.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    fmlalb z8.s, z16.h, z3.h[6]
    fmlalb z9.s, z17.h, z3.h[6]
    fmlalb z10.s, z18.h, z3.h[6]
    fmlalb z11.s, z19.h, z3.h[6]
    fmlalt z8.s, z16.h, z3.h[7]
    fmlalt z9.s, z17.h, z3.h[7]
    fmlalt z10.s, z18.h, z3.h[7]
    fmlalt z11.s, z19.h, z3.h[7]
    fmlalb z12.s, z20.h, z3.h[6]
    fmlalb z13.s, z21.h, z3.h[6]
    fmlalb z14.s, z22.h, z3.h[6]
    fmlalb z15.s, z23.h, z3.h[6]
    fmlalt z12.s, z20.h, z3.h[7]
    fmlalt z13.s, z21.h, z3.h[7]
    fmlalt z14.s, z22.h, z3.h[7]
    fmlalt z15.s, z23.h, z3.h[7]
KAI_ASM_LABEL(label_13)  // Width 2: Multiply loop: multiply skip
    tbz x15, #1, label_14
    add x21, x0, #0x2
    add x20, x0, #0x0
    ld1rh { z6.h }, p1/Z, [x21]
    ld1rh { z22.h }, p1/Z, [x20]
    fcvt z8.h, p1/M, z8.s
    fcvt z18.h, p1/M, z9.s
    uzp1 z18.h, z8.h, z18.h
    fcvt z10.h, p1/M, z10.s
    fcvt z19.h, p1/M, z11.s
    uzp1 z19.h, z10.h, z19.h
    fcvt z12.h, p1/M, z12.s
    fcvt z16.h, p1/M, z13.s
    uzp1 z16.h, z12.h, z16.h
    fcvt z14.h, p1/M, z14.s
    fcvt z17.h, p1/M, z15.s
    uzp1 z17.h, z14.h, z17.h
    FMIN z18.h, p1/M, z18.h, z22.h
    FMAX z18.h, p1/M, z18.h, z6.h
    FMIN z19.h, p1/M, z19.h, z22.h
    FMAX z19.h, p1/M, z19.h, z6.h

    FMIN z16.h, p1/M, z16.h, z22.h
    FMAX z16.h, p1/M, z16.h, z6.h
    FMIN z17.h, p1/M, z17.h, z22.h
    FMAX z17.h, p1/M, z17.h, z6.h

    st1h z18.h, p1, [x13]
    st1h z19.h, p1, [x13, #1, MUL VL]
    addvl x13, x13, #2
    st1h z16.h, p2, [x13]
    st1h z17.h, p3, [x13, #1, MUL VL]
    b label_15
KAI_ASM_LABEL(label_14)  // Width 2: No activation
    mov z28.d, z8.d
    mov z29.d, z9.d
    mov z30.d, z10.d
    mov z31.d, z11.d
    mov z20.d, z12.d
    mov z21.d, z13.d
    mov z22.d, z14.d
    mov z23.d, z15.d
    fcvt z28.h, p1/M, z28.s
    fcvt z26.h, p1/M, z29.s
    uzp1 z26.h, z28.h, z26.h
    fcvt z30.h, p1/M, z30.s
    fcvt z27.h, p1/M, z31.s
    uzp1 z27.h, z30.h, z27.h
    fcvt z20.h, p1/M, z20.s
    fcvt z17.h, p1/M, z21.s
    uzp1 z17.h, z20.h, z17.h
    fcvt z22.h, p1/M, z22.s
    fcvt z25.h, p1/M, z23.s
    uzp1 z25.h, z22.h, z25.h
    st1h z26.h, p1, [x13]
    st1h z27.h, p1, [x13, #1, MUL VL]
    addvl x13, x13, #2
    st1h z17.h, p2, [x13]
    st1h z25.h, p3, [x13, #1, MUL VL]
KAI_ASM_LABEL(label_15)  // Width 2: Output done
    b label_28
KAI_ASM_LABEL(label_16)  // Width 3
    add x26, x14, x12, LSL #2
    cntw x20, ALL, MUL #10
    ld1h z4.s, p1/Z, [x14]
    ld1h z5.s, p1/Z, [x14, #1, MUL VL]
    add x25, x14, x12, LSL #1
    add x24, x26, x12
    cmp x5, x20
    add x23, x14, x12
    ld1h z8.s, p1/Z, [x25]
    ld1h z9.s, p1/Z, [x25, #1, MUL VL]
    add x22, x25, x12
    csel x24, x24, x14, GT
    ld1h z6.s, p1/Z, [x23]
    ld1h z7.s, p1/Z, [x23, #1, MUL VL]
    mov x20, #0x2
    mov x11, x7
    msub x21, x6, x20, x5
    mov x10, x16
    ld1h z10.s, p1/Z, [x22]
    ld1h z11.s, p1/Z, [x22, #1, MUL VL]
    lsl x20, x7, #0x1
    mov x4, xzr
    whilelt p2.h, x4, x21
    inch x4
    whilelt p3.h, x4, x21
    cmp x11, #0x8
    rprfm pldkeep, x20, [x10]
    ld1h z12.s, p1/Z, [x26]
    ld1h z13.s, p1/Z, [x26, #1, MUL VL]
    inch x14, ALL, MUL #2
    inch x23, ALL, MUL #2
    inch x25, ALL, MUL #2
    inch x22, ALL, MUL #2
    ld1h z14.s, p1/Z, [x24]
    ld1h z15.s, p1/Z, [x24, #1, MUL VL]
    fcvt z4.s, p1/M, z4.h
    fcvt z5.s, p1/M, z5.h
    fcvt z6.s, p1/M, z6.h
    fcvt z7.s, p1/M, z7.h
    fcvt z8.s, p1/M, z8.h
    fcvt z9.s, p1/M, z9.h
    fcvt z10.s, p1/M, z10.h
    fcvt z11.s, p1/M, z11.h
    fcvt z12.s, p1/M, z12.h
    fcvt z13.s, p1/M, z13.h
    fcvt z14.s, p1/M, z14.h
    fcvt z15.s, p1/M, z15.h
    inch x26, ALL, MUL #2
    inch x24, ALL, MUL #2
    ble label_18
KAI_ASM_LABEL(label_17)  // Width 3: Multiply loop: Main loop head
    whilelt p0.h, XZR, x11

    ld1rqh { z3.h }, p0/Z, [x10]
    sub x11, x11, #0x8
    add x10, x10, #0x10
    cmp x11, #0x8
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x23]
    ldnt1h z19.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x25]
    ldnt1h z21.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z22.h, p1/Z, [x22]
    ldnt1h z23.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z24.h, p1/Z, [x26]
    ldnt1h z25.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1h z26.h, p1/Z, [x24]
    ldnt1h z27.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z4.s, z16.h, z3.h[0]
    fmlalb z5.s, z17.h, z3.h[0]
    fmlalb z6.s, z18.h, z3.h[0]
    fmlalb z7.s, z19.h, z3.h[0]
    fmlalt z4.s, z16.h, z3.h[1]
    fmlalt z5.s, z17.h, z3.h[1]
    fmlalt z6.s, z18.h, z3.h[1]
    fmlalt z7.s, z19.h, z3.h[1]
    fmlalb z8.s, z20.h, z3.h[0]
    fmlalb z9.s, z21.h, z3.h[0]
    fmlalb z10.s, z22.h, z3.h[0]
    fmlalb z11.s, z23.h, z3.h[0]
    fmlalt z8.s, z20.h, z3.h[1]
    fmlalt z9.s, z21.h, z3.h[1]
    fmlalt z10.s, z22.h, z3.h[1]
    fmlalt z11.s, z23.h, z3.h[1]
    fmlalb z12.s, z24.h, z3.h[0]
    fmlalb z13.s, z25.h, z3.h[0]
    fmlalb z14.s, z26.h, z3.h[0]
    fmlalb z15.s, z27.h, z3.h[0]
    fmlalt z12.s, z24.h, z3.h[1]
    fmlalt z13.s, z25.h, z3.h[1]
    fmlalt z14.s, z26.h, z3.h[1]
    fmlalt z15.s, z27.h, z3.h[1]
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x23]
    ldnt1h z19.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x25]
    ldnt1h z21.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z22.h, p1/Z, [x22]
    ldnt1h z23.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z24.h, p1/Z, [x26]
    ldnt1h z25.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1h z26.h, p1/Z, [x24]
    ldnt1h z27.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z4.s, z16.h, z3.h[2]
    fmlalb z5.s, z17.h, z3.h[2]
    fmlalb z6.s, z18.h, z3.h[2]
    fmlalb z7.s, z19.h, z3.h[2]
    fmlalt z4.s, z16.h, z3.h[3]
    fmlalt z5.s, z17.h, z3.h[3]
    fmlalt z6.s, z18.h, z3.h[3]
    fmlalt z7.s, z19.h, z3.h[3]
    fmlalb z8.s, z20.h, z3.h[2]
    fmlalb z9.s, z21.h, z3.h[2]
    fmlalb z10.s, z22.h, z3.h[2]
    fmlalb z11.s, z23.h, z3.h[2]
    fmlalt z8.s, z20.h, z3.h[3]
    fmlalt z9.s, z21.h, z3.h[3]
    fmlalt z10.s, z22.h, z3.h[3]
    fmlalt z11.s, z23.h, z3.h[3]
    fmlalb z12.s, z24.h, z3.h[2]
    fmlalb z13.s, z25.h, z3.h[2]
    fmlalb z14.s, z26.h, z3.h[2]
    fmlalb z15.s, z27.h, z3.h[2]
    fmlalt z12.s, z24.h, z3.h[3]
    fmlalt z13.s, z25.h, z3.h[3]
    fmlalt z14.s, z26.h, z3.h[3]
    fmlalt z15.s, z27.h, z3.h[3]
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x23]
    ldnt1h z19.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x25]
    ldnt1h z21.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z22.h, p1/Z, [x22]
    ldnt1h z23.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z24.h, p1/Z, [x26]
    ldnt1h z25.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1h z26.h, p1/Z, [x24]
    ldnt1h z27.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z4.s, z16.h, z3.h[4]
    fmlalb z5.s, z17.h, z3.h[4]
    fmlalb z6.s, z18.h, z3.h[4]
    fmlalb z7.s, z19.h, z3.h[4]
    fmlalt z4.s, z16.h, z3.h[5]
    fmlalt z5.s, z17.h, z3.h[5]
    fmlalt z6.s, z18.h, z3.h[5]
    fmlalt z7.s, z19.h, z3.h[5]
    fmlalb z8.s, z20.h, z3.h[4]
    fmlalb z9.s, z21.h, z3.h[4]
    fmlalb z10.s, z22.h, z3.h[4]
    fmlalb z11.s, z23.h, z3.h[4]
    fmlalt z8.s, z20.h, z3.h[5]
    fmlalt z9.s, z21.h, z3.h[5]
    fmlalt z10.s, z22.h, z3.h[5]
    fmlalt z11.s, z23.h, z3.h[5]
    fmlalb z12.s, z24.h, z3.h[4]
    fmlalb z13.s, z25.h, z3.h[4]
    fmlalb z14.s, z26.h, z3.h[4]
    fmlalb z15.s, z27.h, z3.h[4]
    fmlalt z12.s, z24.h, z3.h[5]
    fmlalt z13.s, z25.h, z3.h[5]
    fmlalt z14.s, z26.h, z3.h[5]
    fmlalt z15.s, z27.h, z3.h[5]
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x23]
    ldnt1h z19.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x25]
    ldnt1h z21.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z22.h, p1/Z, [x22]
    ldnt1h z23.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z24.h, p1/Z, [x26]
    ldnt1h z25.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1h z26.h, p1/Z, [x24]
    ldnt1h z27.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z4.s, z16.h, z3.h[6]
    fmlalb z5.s, z17.h, z3.h[6]
    fmlalb z6.s, z18.h, z3.h[6]
    fmlalb z7.s, z19.h, z3.h[6]
    fmlalt z4.s, z16.h, z3.h[7]
    fmlalt z5.s, z17.h, z3.h[7]
    fmlalt z6.s, z18.h, z3.h[7]
    fmlalt z7.s, z19.h, z3.h[7]
    fmlalb z8.s, z20.h, z3.h[6]
    fmlalb z9.s, z21.h, z3.h[6]
    fmlalb z10.s, z22.h, z3.h[6]
    fmlalb z11.s, z23.h, z3.h[6]
    fmlalt z8.s, z20.h, z3.h[7]
    fmlalt z9.s, z21.h, z3.h[7]
    fmlalt z10.s, z22.h, z3.h[7]
    fmlalt z11.s, z23.h, z3.h[7]
    fmlalb z12.s, z24.h, z3.h[6]
    fmlalb z13.s, z25.h, z3.h[6]
    fmlalb z14.s, z26.h, z3.h[6]
    fmlalb z15.s, z27.h, z3.h[6]
    fmlalt z12.s, z24.h, z3.h[7]
    fmlalt z13.s, z25.h, z3.h[7]
    fmlalt z14.s, z26.h, z3.h[7]
    fmlalt z15.s, z27.h, z3.h[7]
    bgt label_17
KAI_ASM_LABEL(label_18)  // Width 3: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x11
    subs x11, x11, #0x2
    ld1rqh { z3.h }, p0/Z, [x10]
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x23]
    ldnt1h z19.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x25]
    ldnt1h z21.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z22.h, p1/Z, [x22]
    ldnt1h z23.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z24.h, p1/Z, [x26]
    ldnt1h z25.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1h z26.h, p1/Z, [x24]
    ldnt1h z27.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z4.s, z16.h, z3.h[0]
    fmlalb z5.s, z17.h, z3.h[0]
    fmlalb z6.s, z18.h, z3.h[0]
    fmlalb z7.s, z19.h, z3.h[0]
    fmlalt z4.s, z16.h, z3.h[1]
    fmlalt z5.s, z17.h, z3.h[1]
    fmlalt z6.s, z18.h, z3.h[1]
    fmlalt z7.s, z19.h, z3.h[1]
    fmlalb z8.s, z20.h, z3.h[0]
    fmlalb z9.s, z21.h, z3.h[0]
    fmlalb z10.s, z22.h, z3.h[0]
    fmlalb z11.s, z23.h, z3.h[0]
    fmlalt z8.s, z20.h, z3.h[1]
    fmlalt z9.s, z21.h, z3.h[1]
    fmlalt z10.s, z22.h, z3.h[1]
    fmlalt z11.s, z23.h, z3.h[1]
    fmlalb z12.s, z24.h, z3.h[0]
    fmlalb z13.s, z25.h, z3.h[0]
    fmlalb z14.s, z26.h, z3.h[0]
    fmlalb z15.s, z27.h, z3.h[0]
    fmlalt z12.s, z24.h, z3.h[1]
    fmlalt z13.s, z25.h, z3.h[1]
    fmlalt z14.s, z26.h, z3.h[1]
    fmlalt z15.s, z27.h, z3.h[1]
    ble label_19
    subs x11, x11, #0x2
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x23]
    ldnt1h z19.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x25]
    ldnt1h z21.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z22.h, p1/Z, [x22]
    ldnt1h z23.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z24.h, p1/Z, [x26]
    ldnt1h z25.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1h z26.h, p1/Z, [x24]
    ldnt1h z27.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z4.s, z16.h, z3.h[2]
    fmlalb z5.s, z17.h, z3.h[2]
    fmlalb z6.s, z18.h, z3.h[2]
    fmlalb z7.s, z19.h, z3.h[2]
    fmlalt z4.s, z16.h, z3.h[3]
    fmlalt z5.s, z17.h, z3.h[3]
    fmlalt z6.s, z18.h, z3.h[3]
    fmlalt z7.s, z19.h, z3.h[3]
    fmlalb z8.s, z20.h, z3.h[2]
    fmlalb z9.s, z21.h, z3.h[2]
    fmlalb z10.s, z22.h, z3.h[2]
    fmlalb z11.s, z23.h, z3.h[2]
    fmlalt z8.s, z20.h, z3.h[3]
    fmlalt z9.s, z21.h, z3.h[3]
    fmlalt z10.s, z22.h, z3.h[3]
    fmlalt z11.s, z23.h, z3.h[3]
    fmlalb z12.s, z24.h, z3.h[2]
    fmlalb z13.s, z25.h, z3.h[2]
    fmlalb z14.s, z26.h, z3.h[2]
    fmlalb z15.s, z27.h, z3.h[2]
    fmlalt z12.s, z24.h, z3.h[3]
    fmlalt z13.s, z25.h, z3.h[3]
    fmlalt z14.s, z26.h, z3.h[3]
    fmlalt z15.s, z27.h, z3.h[3]
    ble label_19
    subs x11, x11, #0x2
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x23]
    ldnt1h z19.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x25]
    ldnt1h z21.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z22.h, p1/Z, [x22]
    ldnt1h z23.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z24.h, p1/Z, [x26]
    ldnt1h z25.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1h z26.h, p1/Z, [x24]
    ldnt1h z27.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z4.s, z16.h, z3.h[4]
    fmlalb z5.s, z17.h, z3.h[4]
    fmlalb z6.s, z18.h, z3.h[4]
    fmlalb z7.s, z19.h, z3.h[4]
    fmlalt z4.s, z16.h, z3.h[5]
    fmlalt z5.s, z17.h, z3.h[5]
    fmlalt z6.s, z18.h, z3.h[5]
    fmlalt z7.s, z19.h, z3.h[5]
    fmlalb z8.s, z20.h, z3.h[4]
    fmlalb z9.s, z21.h, z3.h[4]
    fmlalb z10.s, z22.h, z3.h[4]
    fmlalb z11.s, z23.h, z3.h[4]
    fmlalt z8.s, z20.h, z3.h[5]
    fmlalt z9.s, z21.h, z3.h[5]
    fmlalt z10.s, z22.h, z3.h[5]
    fmlalt z11.s, z23.h, z3.h[5]
    fmlalb z12.s, z24.h, z3.h[4]
    fmlalb z13.s, z25.h, z3.h[4]
    fmlalb z14.s, z26.h, z3.h[4]
    fmlalb z15.s, z27.h, z3.h[4]
    fmlalt z12.s, z24.h, z3.h[5]
    fmlalt z13.s, z25.h, z3.h[5]
    fmlalt z14.s, z26.h, z3.h[5]
    fmlalt z15.s, z27.h, z3.h[5]
    ble label_19
    ldnt1h z16.h, p1/Z, [x14]
    ldnt1h z17.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z18.h, p1/Z, [x23]
    ldnt1h z19.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x25]
    ldnt1h z21.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z22.h, p1/Z, [x22]
    ldnt1h z23.h, p1/Z, [x22, #1, MUL VL]
    addvl x22, x22, #2
    ldnt1h z24.h, p1/Z, [x26]
    ldnt1h z25.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    ldnt1h z26.h, p1/Z, [x24]
    ldnt1h z27.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z4.s, z16.h, z3.h[6]
    fmlalb z5.s, z17.h, z3.h[6]
    fmlalb z6.s, z18.h, z3.h[6]
    fmlalb z7.s, z19.h, z3.h[6]
    fmlalt z4.s, z16.h, z3.h[7]
    fmlalt z5.s, z17.h, z3.h[7]
    fmlalt z6.s, z18.h, z3.h[7]
    fmlalt z7.s, z19.h, z3.h[7]
    fmlalb z8.s, z20.h, z3.h[6]
    fmlalb z9.s, z21.h, z3.h[6]
    fmlalb z10.s, z22.h, z3.h[6]
    fmlalb z11.s, z23.h, z3.h[6]
    fmlalt z8.s, z20.h, z3.h[7]
    fmlalt z9.s, z21.h, z3.h[7]
    fmlalt z10.s, z22.h, z3.h[7]
    fmlalt z11.s, z23.h, z3.h[7]
    fmlalb z12.s, z24.h, z3.h[6]
    fmlalb z13.s, z25.h, z3.h[6]
    fmlalb z14.s, z26.h, z3.h[6]
    fmlalb z15.s, z27.h, z3.h[6]
    fmlalt z12.s, z24.h, z3.h[7]
    fmlalt z13.s, z25.h, z3.h[7]
    fmlalt z14.s, z26.h, z3.h[7]
    fmlalt z15.s, z27.h, z3.h[7]
KAI_ASM_LABEL(label_19)  // Width 3: Multiply loop: multiply skip
    tbz x15, #1, label_20
    mov z24.d, z4.d
    mov z25.d, z5.d
    mov z26.d, z6.d
    mov z27.d, z7.d
    mov z28.d, z8.d
    mov z29.d, z9.d
    mov z30.d, z10.d
    mov z31.d, z11.d
    mov z0.d, z12.d
    mov z1.d, z13.d
    mov z2.d, z14.d
    mov z24.d, z15.d
    add x21, x0, #0x2
    add x20, x0, #0x0
    ld1rh { z19.h }, p1/Z, [x21]
    ld1rh { z18.h }, p1/Z, [x20]
    fcvt z24.h, p1/M, z24.s
    fcvt z20.h, p1/M, z25.s
    uzp1 z20.h, z24.h, z20.h
    fcvt z26.h, p1/M, z26.s
    fcvt z21.h, p1/M, z27.s
    uzp1 z21.h, z26.h, z21.h
    fcvt z28.h, p1/M, z28.s
    fcvt z14.h, p1/M, z29.s
    uzp1 z14.h, z28.h, z14.h
    fcvt z30.h, p1/M, z30.s
    fcvt z15.h, p1/M, z31.s
    uzp1 z15.h, z30.h, z15.h
    fcvt z0.h, p1/M, z0.s
    fcvt z16.h, p1/M, z1.s
    uzp1 z16.h, z0.h, z16.h
    fcvt z2.h, p1/M, z2.s
    fcvt z17.h, p1/M, z3.s
    uzp1 z17.h, z2.h, z17.h
    FMIN z20.h, p1/M, z20.h, z18.h
    FMAX z20.h, p1/M, z20.h, z19.h
    FMIN z21.h, p1/M, z21.h, z18.h
    FMAX z21.h, p1/M, z21.h, z19.h

    FMIN z14.h, p1/M, z14.h, z18.h
    FMAX z14.h, p1/M, z14.h, z19.h
    FMIN z15.h, p1/M, z15.h, z18.h
    FMAX z15.h, p1/M, z15.h, z19.h

    FMIN z16.h, p1/M, z16.h, z18.h
    FMAX z16.h, p1/M, z16.h, z19.h
    FMIN z17.h, p1/M, z17.h, z18.h
    FMAX z17.h, p1/M, z17.h, z19.h

    st1h z20.h, p1, [x13]
    st1h z21.h, p1, [x13, #1, MUL VL]
    addvl x13, x13, #2
    st1h z14.h, p1, [x13]
    st1h z15.h, p1, [x13, #1, MUL VL]
    addvl x13, x13, #2
    st1h z16.h, p2, [x13]
    st1h z17.h, p3, [x13, #1, MUL VL]
    b label_21
KAI_ASM_LABEL(label_20)  // Width 3: No activation
    fcvt z4.h, p1/M, z4.s
    fcvt z17.h, p1/M, z5.s
    uzp1 z17.h, z4.h, z17.h
    fcvt z6.h, p1/M, z6.s
    fcvt z25.h, p1/M, z7.s
    uzp1 z25.h, z6.h, z25.h
    fcvt z8.h, p1/M, z8.s
    fcvt z18.h, p1/M, z9.s
    uzp1 z18.h, z8.h, z18.h
    fcvt z10.h, p1/M, z10.s
    fcvt z19.h, p1/M, z11.s
    uzp1 z19.h, z10.h, z19.h
    fcvt z12.h, p1/M, z12.s
    fcvt z17.h, p1/M, z13.s
    uzp1 z17.h, z12.h, z17.h
    fcvt z14.h, p1/M, z14.s
    fcvt z25.h, p1/M, z15.s
    uzp1 z25.h, z14.h, z25.h
    st1h z17.h, p1, [x13]
    st1h z25.h, p1, [x13, #1, MUL VL]
    addvl x13, x13, #2
    st1h z18.h, p1, [x13]
    st1h z19.h, p1, [x13, #1, MUL VL]
    addvl x13, x13, #2
    st1h z17.h, p2, [x13]
    st1h z25.h, p3, [x13, #1, MUL VL]
KAI_ASM_LABEL(label_21)  // Width 3: Output done
    b label_28
KAI_ASM_LABEL(label_22)  // Width 4
    add x9, x14, x12, LSL #2
    cntw x20, ALL, MUL #14
    ld1h { z12.s }, p1/Z, [x14]
    add x28, x9, x12, LSL #1
    add x27, x14, x12, LSL #1
    ld1h { z13.s }, p1/Z, [x14, #1, MUL VL]
    add x26, x28, x12
    cmp x5, x20
    ld1h { z16.s }, p1/Z, [x27]
    add x25, x14, x12
    add x24, x27, x12
    ld1h { z17.s }, p1/Z, [x27, #1, MUL VL]
    fcvt z12.s, p1/m, z12.h
    add x23, x9, x12
    csel x26, x26, x14, GT
    ld1h { z14.s }, p1/Z, [x25]
    fcvt z13.s, p1/m, z13.h
    ld1h { z15.s }, p1/Z, [x25, #1, MUL VL]
    fcvt z16.s, p1/m, z16.h
    mov x20, #0x3
    mov x11, x7
    ld1h { z18.s }, p1/Z, [x24]
    fcvt z17.s, p1/m, z17.h
    msub x21, x6, x20, x5
    mov x10, x16
    ld1h { z19.s }, p1/Z, [x24, #1, MUL VL]
    fcvt z14.s, p1/m, z14.h
    lsl x20, x7, #0x1
    mov x4,XZR
    whilelt p2.h, x4, x21
    inch x4
    whilelt p3.h, x4, x21
    ld1h { z28.s }, p1/Z, [x9]
    fcvt z15.s, p1/m, z15.h
    cmp x11, #0x8
    rprfm pldkeep, x20, [x10]
    ld1h { z29.s }, p1/Z, [x9, #1, MUL VL]
    fcvt z18.s, p1/m, z18.h
    add x22, x14, x12, LSL #3
    inch x14, ALL, MUL #2
    ld1h { z30.s }, p1/Z, [x23]
    fcvt z19.s, p1/m, z19.h
    inch x25, ALL, MUL #2
    inch x27, ALL, MUL #2
    ld1h { z31.s }, p1/Z, [x23, #1, MUL VL]
    fcvt z28.s, p1/m, z28.h
    inch x24, ALL, MUL #2
    inch x9, ALL, MUL #2
    ld1h { z8.s }, p1/Z, [x28]
    fcvt z29.s, p1/m, z29.h
    inch x23, ALL, MUL #2
    ld1h { z9.s }, p1/Z, [x28, #1, MUL VL]
    fcvt z30.s, p1/m, z30.h
    inch x28, ALL, MUL #2
    ld1h { z10.s }, p1/Z, [x26]
    fcvt z31.s, p1/m, z31.h
    ld1h { z11.s }, p1/Z, [x26, #1, MUL VL]
    fcvt z8.s, p1/m, z8.h
    inch x26, ALL, MUL #2
    fcvt z9.s, p1/m, z9.h
    fcvt z10.s, p1/m, z10.h
    fcvt z11.s, p1/m, z11.h
    ble label_24
KAI_ASM_LABEL(label_23)  // Width 4: Multiply loop: Main loop head
    whilelt p0.h, XZR, x11
    ld1rqh { z3.h }, p0/Z, [x10]
    sub x11, x11, #0x8
    add x10, x10, #0x10
    cmp x11, #0x8
    ldnt1h z4.h, p1/Z, [x14]
    ldnt1h z5.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z6.h, p1/Z, [x25]
    ldnt1h z7.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z20.h, p1/Z, [x27]
    ldnt1h z21.h, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1h z22.h, p1/Z, [x24]
    ldnt1h z23.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z12.s, z4.h, z3.h[0]
    fmlalb z13.s, z5.h, z3.h[0]
    fmlalb z14.s, z6.h, z3.h[0]
    fmlalb z15.s, z7.h, z3.h[0]
    fmlalt z12.s, z4.h, z3.h[1]
    fmlalt z13.s, z5.h, z3.h[1]
    fmlalt z14.s, z6.h, z3.h[1]
    fmlalt z15.s, z7.h, z3.h[1]
    fmlalb z16.s, z20.h, z3.h[0]
    fmlalb z17.s, z21.h, z3.h[0]
    fmlalb z18.s, z22.h, z3.h[0]
    fmlalb z19.s, z23.h, z3.h[0]
    fmlalt z16.s, z20.h, z3.h[1]
    fmlalt z17.s, z21.h, z3.h[1]
    fmlalt z18.s, z22.h, z3.h[1]
    fmlalt z19.s, z23.h, z3.h[1]
    ldnt1h z4.h, p1/Z, [x9]
    ldnt1h z5.h, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1h z6.h, p1/Z, [x23]
    ldnt1h z7.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x28]
    ldnt1h z21.h, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1h z22.h, p1/Z, [x26]
    ldnt1h z23.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmlalb z28.s, z4.h, z3.h[0]
    fmlalb z29.s, z5.h, z3.h[0]
    fmlalb z30.s, z6.h, z3.h[0]
    fmlalb z31.s, z7.h, z3.h[0]
    fmlalt z28.s, z4.h, z3.h[1]
    fmlalt z29.s, z5.h, z3.h[1]
    fmlalt z30.s, z6.h, z3.h[1]
    fmlalt z31.s, z7.h, z3.h[1]
    fmlalb z8.s, z20.h, z3.h[0]
    fmlalb z9.s, z21.h, z3.h[0]
    fmlalb z10.s, z22.h, z3.h[0]
    fmlalb z11.s, z23.h, z3.h[0]
    fmlalt z8.s, z20.h, z3.h[1]
    fmlalt z9.s, z21.h, z3.h[1]
    fmlalt z10.s, z22.h, z3.h[1]
    fmlalt z11.s, z23.h, z3.h[1]
    ldnt1h z4.h, p1/Z, [x14]
    ldnt1h z5.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z6.h, p1/Z, [x25]
    ldnt1h z7.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z20.h, p1/Z, [x27]
    ldnt1h z21.h, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1h z22.h, p1/Z, [x24]
    ldnt1h z23.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z12.s, z4.h, z3.h[2]
    fmlalb z13.s, z5.h, z3.h[2]
    fmlalb z14.s, z6.h, z3.h[2]
    fmlalb z15.s, z7.h, z3.h[2]
    fmlalt z12.s, z4.h, z3.h[3]
    fmlalt z13.s, z5.h, z3.h[3]
    fmlalt z14.s, z6.h, z3.h[3]
    fmlalt z15.s, z7.h, z3.h[3]
    fmlalb z16.s, z20.h, z3.h[2]
    fmlalb z17.s, z21.h, z3.h[2]
    fmlalb z18.s, z22.h, z3.h[2]
    fmlalb z19.s, z23.h, z3.h[2]
    fmlalt z16.s, z20.h, z3.h[3]
    fmlalt z17.s, z21.h, z3.h[3]
    fmlalt z18.s, z22.h, z3.h[3]
    fmlalt z19.s, z23.h, z3.h[3]
    ldnt1h z4.h, p1/Z, [x9]
    ldnt1h z5.h, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1h z6.h, p1/Z, [x23]
    ldnt1h z7.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x28]
    ldnt1h z21.h, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1h z22.h, p1/Z, [x26]
    ldnt1h z23.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmlalb z28.s, z4.h, z3.h[2]
    fmlalb z29.s, z5.h, z3.h[2]
    fmlalb z30.s, z6.h, z3.h[2]
    fmlalb z31.s, z7.h, z3.h[2]
    fmlalt z28.s, z4.h, z3.h[3]
    fmlalt z29.s, z5.h, z3.h[3]
    fmlalt z30.s, z6.h, z3.h[3]
    fmlalt z31.s, z7.h, z3.h[3]
    fmlalb z8.s, z20.h, z3.h[2]
    fmlalb z9.s, z21.h, z3.h[2]
    fmlalb z10.s, z22.h, z3.h[2]
    fmlalb z11.s, z23.h, z3.h[2]
    fmlalt z8.s, z20.h, z3.h[3]
    fmlalt z9.s, z21.h, z3.h[3]
    fmlalt z10.s, z22.h, z3.h[3]
    fmlalt z11.s, z23.h, z3.h[3]
    ldnt1h z4.h, p1/Z, [x14]
    ldnt1h z5.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z6.h, p1/Z, [x25]
    ldnt1h z7.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z20.h, p1/Z, [x27]
    ldnt1h z21.h, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1h z22.h, p1/Z, [x24]
    ldnt1h z23.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z12.s, z4.h, z3.h[4]
    fmlalb z13.s, z5.h, z3.h[4]
    fmlalb z14.s, z6.h, z3.h[4]
    fmlalb z15.s, z7.h, z3.h[4]
    fmlalt z12.s, z4.h, z3.h[5]
    fmlalt z13.s, z5.h, z3.h[5]
    fmlalt z14.s, z6.h, z3.h[5]
    fmlalt z15.s, z7.h, z3.h[5]
    fmlalb z16.s, z20.h, z3.h[4]
    fmlalb z17.s, z21.h, z3.h[4]
    fmlalb z18.s, z22.h, z3.h[4]
    fmlalb z19.s, z23.h, z3.h[4]
    fmlalt z16.s, z20.h, z3.h[5]
    fmlalt z17.s, z21.h, z3.h[5]
    fmlalt z18.s, z22.h, z3.h[5]
    fmlalt z19.s, z23.h, z3.h[5]
    ldnt1h z4.h, p1/Z, [x9]
    ldnt1h z5.h, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1h z6.h, p1/Z, [x23]
    ldnt1h z7.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x28]
    ldnt1h z21.h, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1h z22.h, p1/Z, [x26]
    ldnt1h z23.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmlalb z28.s, z4.h, z3.h[4]
    fmlalb z29.s, z5.h, z3.h[4]
    fmlalb z30.s, z6.h, z3.h[4]
    fmlalb z31.s, z7.h, z3.h[4]
    fmlalt z28.s, z4.h, z3.h[5]
    fmlalt z29.s, z5.h, z3.h[5]
    fmlalt z30.s, z6.h, z3.h[5]
    fmlalt z31.s, z7.h, z3.h[5]
    fmlalb z8.s, z20.h, z3.h[4]
    fmlalb z9.s, z21.h, z3.h[4]
    fmlalb z10.s, z22.h, z3.h[4]
    fmlalb z11.s, z23.h, z3.h[4]
    fmlalt z8.s, z20.h, z3.h[5]
    fmlalt z9.s, z21.h, z3.h[5]
    fmlalt z10.s, z22.h, z3.h[5]
    fmlalt z11.s, z23.h, z3.h[5]
    ldnt1h z4.h, p1/Z, [x14]
    ldnt1h z5.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z6.h, p1/Z, [x25]
    ldnt1h z7.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z20.h, p1/Z, [x27]
    ldnt1h z21.h, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1h z22.h, p1/Z, [x24]
    ldnt1h z23.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z12.s, z4.h, z3.h[6]
    fmlalb z13.s, z5.h, z3.h[6]
    fmlalb z14.s, z6.h, z3.h[6]
    fmlalb z15.s, z7.h, z3.h[6]
    fmlalt z12.s, z4.h, z3.h[7]
    fmlalt z13.s, z5.h, z3.h[7]
    fmlalt z14.s, z6.h, z3.h[7]
    fmlalt z15.s, z7.h, z3.h[7]
    fmlalb z16.s, z20.h, z3.h[6]
    fmlalb z17.s, z21.h, z3.h[6]
    fmlalb z18.s, z22.h, z3.h[6]
    fmlalb z19.s, z23.h, z3.h[6]
    fmlalt z16.s, z20.h, z3.h[7]
    fmlalt z17.s, z21.h, z3.h[7]
    fmlalt z18.s, z22.h, z3.h[7]
    fmlalt z19.s, z23.h, z3.h[7]
    ldnt1h z4.h, p1/Z, [x9]
    ldnt1h z5.h, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1h z6.h, p1/Z, [x23]
    ldnt1h z7.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x28]
    ldnt1h z21.h, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1h z22.h, p1/Z, [x26]
    ldnt1h z23.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmlalb z28.s, z4.h, z3.h[6]
    fmlalb z29.s, z5.h, z3.h[6]
    fmlalb z30.s, z6.h, z3.h[6]
    fmlalb z31.s, z7.h, z3.h[6]
    fmlalt z28.s, z4.h, z3.h[7]
    fmlalt z29.s, z5.h, z3.h[7]
    fmlalt z30.s, z6.h, z3.h[7]
    fmlalt z31.s, z7.h, z3.h[7]
    fmlalb z8.s, z20.h, z3.h[6]
    fmlalb z9.s, z21.h, z3.h[6]
    fmlalb z10.s, z22.h, z3.h[6]
    fmlalb z11.s, z23.h, z3.h[6]
    fmlalt z8.s, z20.h, z3.h[7]
    fmlalt z9.s, z21.h, z3.h[7]
    fmlalt z10.s, z22.h, z3.h[7]
    fmlalt z11.s, z23.h, z3.h[7]
    bgt label_23
KAI_ASM_LABEL(label_24)  // Width 4: Multiply loop: Single iteration only
    whilelt p0.h, XZR, x11
    subs x11, x11, #0x2
    ld1rqh { z3.h }, p0/Z, [x10]
    ldnt1h z4.h, p1/Z, [x14]
    ldnt1h z5.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z6.h, p1/Z, [x25]
    ldnt1h z7.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z20.h, p1/Z, [x27]
    ldnt1h z21.h, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1h z22.h, p1/Z, [x24]
    ldnt1h z23.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z12.s, z4.h, z3.h[0]
    fmlalb z13.s, z5.h, z3.h[0]
    fmlalb z14.s, z6.h, z3.h[0]
    fmlalb z15.s, z7.h, z3.h[0]
    fmlalt z12.s, z4.h, z3.h[1]
    fmlalt z13.s, z5.h, z3.h[1]
    fmlalt z14.s, z6.h, z3.h[1]
    fmlalt z15.s, z7.h, z3.h[1]
    fmlalb z16.s, z20.h, z3.h[0]
    fmlalb z17.s, z21.h, z3.h[0]
    fmlalb z18.s, z22.h, z3.h[0]
    fmlalb z19.s, z23.h, z3.h[0]
    fmlalt z16.s, z20.h, z3.h[1]
    fmlalt z17.s, z21.h, z3.h[1]
    fmlalt z18.s, z22.h, z3.h[1]
    fmlalt z19.s, z23.h, z3.h[1]
    ldnt1h z4.h, p1/Z, [x9]
    ldnt1h z5.h, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1h z6.h, p1/Z, [x23]
    ldnt1h z7.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x28]
    ldnt1h z21.h, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1h z22.h, p1/Z, [x26]
    ldnt1h z23.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmlalb z28.s, z4.h, z3.h[0]
    fmlalb z29.s, z5.h, z3.h[0]
    fmlalb z30.s, z6.h, z3.h[0]
    fmlalb z31.s, z7.h, z3.h[0]
    fmlalt z28.s, z4.h, z3.h[1]
    fmlalt z29.s, z5.h, z3.h[1]
    fmlalt z30.s, z6.h, z3.h[1]
    fmlalt z31.s, z7.h, z3.h[1]
    fmlalb z8.s, z20.h, z3.h[0]
    fmlalb z9.s, z21.h, z3.h[0]
    fmlalb z10.s, z22.h, z3.h[0]
    fmlalb z11.s, z23.h, z3.h[0]
    fmlalt z8.s, z20.h, z3.h[1]
    fmlalt z9.s, z21.h, z3.h[1]
    fmlalt z10.s, z22.h, z3.h[1]
    fmlalt z11.s, z23.h, z3.h[1]
    ble label_25
    subs x11, x11, #0x2
    ldnt1h z4.h, p1/Z, [x14]
    ldnt1h z5.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z6.h, p1/Z, [x25]
    ldnt1h z7.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z20.h, p1/Z, [x27]
    ldnt1h z21.h, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1h z22.h, p1/Z, [x24]
    ldnt1h z23.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z12.s, z4.h, z3.h[2]
    fmlalb z13.s, z5.h, z3.h[2]
    fmlalb z14.s, z6.h, z3.h[2]
    fmlalb z15.s, z7.h, z3.h[2]
    fmlalt z12.s, z4.h, z3.h[3]
    fmlalt z13.s, z5.h, z3.h[3]
    fmlalt z14.s, z6.h, z3.h[3]
    fmlalt z15.s, z7.h, z3.h[3]
    fmlalb z16.s, z20.h, z3.h[2]
    fmlalb z17.s, z21.h, z3.h[2]
    fmlalb z18.s, z22.h, z3.h[2]
    fmlalb z19.s, z23.h, z3.h[2]
    fmlalt z16.s, z20.h, z3.h[3]
    fmlalt z17.s, z21.h, z3.h[3]
    fmlalt z18.s, z22.h, z3.h[3]
    fmlalt z19.s, z23.h, z3.h[3]
    ldnt1h z4.h, p1/Z, [x9]
    ldnt1h z5.h, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1h z6.h, p1/Z, [x23]
    ldnt1h z7.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x28]
    ldnt1h z21.h, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1h z22.h, p1/Z, [x26]
    ldnt1h z23.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmlalb z28.s, z4.h, z3.h[2]
    fmlalb z29.s, z5.h, z3.h[2]
    fmlalb z30.s, z6.h, z3.h[2]
    fmlalb z31.s, z7.h, z3.h[2]
    fmlalt z28.s, z4.h, z3.h[3]
    fmlalt z29.s, z5.h, z3.h[3]
    fmlalt z30.s, z6.h, z3.h[3]
    fmlalt z31.s, z7.h, z3.h[3]
    fmlalb z8.s, z20.h, z3.h[2]
    fmlalb z9.s, z21.h, z3.h[2]
    fmlalb z10.s, z22.h, z3.h[2]
    fmlalb z11.s, z23.h, z3.h[2]
    fmlalt z8.s, z20.h, z3.h[3]
    fmlalt z9.s, z21.h, z3.h[3]
    fmlalt z10.s, z22.h, z3.h[3]
    fmlalt z11.s, z23.h, z3.h[3]
    ble label_25
    subs x11, x11, #0x2
    ldnt1h z4.h, p1/Z, [x14]
    ldnt1h z5.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z6.h, p1/Z, [x25]
    ldnt1h z7.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z20.h, p1/Z, [x27]
    ldnt1h z21.h, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1h z22.h, p1/Z, [x24]
    ldnt1h z23.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z12.s, z4.h, z3.h[4]
    fmlalb z13.s, z5.h, z3.h[4]
    fmlalb z14.s, z6.h, z3.h[4]
    fmlalb z15.s, z7.h, z3.h[4]
    fmlalt z12.s, z4.h, z3.h[5]
    fmlalt z13.s, z5.h, z3.h[5]
    fmlalt z14.s, z6.h, z3.h[5]
    fmlalt z15.s, z7.h, z3.h[5]
    fmlalb z16.s, z20.h, z3.h[4]
    fmlalb z17.s, z21.h, z3.h[4]
    fmlalb z18.s, z22.h, z3.h[4]
    fmlalb z19.s, z23.h, z3.h[4]
    fmlalt z16.s, z20.h, z3.h[5]
    fmlalt z17.s, z21.h, z3.h[5]
    fmlalt z18.s, z22.h, z3.h[5]
    fmlalt z19.s, z23.h, z3.h[5]
    ldnt1h z4.h, p1/Z, [x9]
    ldnt1h z5.h, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1h z6.h, p1/Z, [x23]
    ldnt1h z7.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x28]
    ldnt1h z21.h, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1h z22.h, p1/Z, [x26]
    ldnt1h z23.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmlalb z28.s, z4.h, z3.h[4]
    fmlalb z29.s, z5.h, z3.h[4]
    fmlalb z30.s, z6.h, z3.h[4]
    fmlalb z31.s, z7.h, z3.h[4]
    fmlalt z28.s, z4.h, z3.h[5]
    fmlalt z29.s, z5.h, z3.h[5]
    fmlalt z30.s, z6.h, z3.h[5]
    fmlalt z31.s, z7.h, z3.h[5]
    fmlalb z8.s, z20.h, z3.h[4]
    fmlalb z9.s, z21.h, z3.h[4]
    fmlalb z10.s, z22.h, z3.h[4]
    fmlalb z11.s, z23.h, z3.h[4]
    fmlalt z8.s, z20.h, z3.h[5]
    fmlalt z9.s, z21.h, z3.h[5]
    fmlalt z10.s, z22.h, z3.h[5]
    fmlalt z11.s, z23.h, z3.h[5]
    ble label_25
    ldnt1h z4.h, p1/Z, [x14]
    ldnt1h z5.h, p1/Z, [x14, #1, MUL VL]
    addvl x14, x14, #2
    ldnt1h z6.h, p1/Z, [x25]
    ldnt1h z7.h, p1/Z, [x25, #1, MUL VL]
    addvl x25, x25, #2
    ldnt1h z20.h, p1/Z, [x27]
    ldnt1h z21.h, p1/Z, [x27, #1, MUL VL]
    addvl x27, x27, #2
    ldnt1h z22.h, p1/Z, [x24]
    ldnt1h z23.h, p1/Z, [x24, #1, MUL VL]
    addvl x24, x24, #2
    fmlalb z12.s, z4.h, z3.h[6]
    fmlalb z13.s, z5.h, z3.h[6]
    fmlalb z14.s, z6.h, z3.h[6]
    fmlalb z15.s, z7.h, z3.h[6]
    fmlalt z12.s, z4.h, z3.h[7]
    fmlalt z13.s, z5.h, z3.h[7]
    fmlalt z14.s, z6.h, z3.h[7]
    fmlalt z15.s, z7.h, z3.h[7]
    fmlalb z16.s, z20.h, z3.h[6]
    fmlalb z17.s, z21.h, z3.h[6]
    fmlalb z18.s, z22.h, z3.h[6]
    fmlalb z19.s, z23.h, z3.h[6]
    fmlalt z16.s, z20.h, z3.h[7]
    fmlalt z17.s, z21.h, z3.h[7]
    fmlalt z18.s, z22.h, z3.h[7]
    fmlalt z19.s, z23.h, z3.h[7]
    ldnt1h z4.h, p1/Z, [x9]
    ldnt1h z5.h, p1/Z, [x9, #1, MUL VL]
    addvl x9, x9, #2
    ldnt1h z6.h, p1/Z, [x23]
    ldnt1h z7.h, p1/Z, [x23, #1, MUL VL]
    addvl x23, x23, #2
    ldnt1h z20.h, p1/Z, [x28]
    ldnt1h z21.h, p1/Z, [x28, #1, MUL VL]
    addvl x28, x28, #2
    ldnt1h z22.h, p1/Z, [x26]
    ldnt1h z23.h, p1/Z, [x26, #1, MUL VL]
    addvl x26, x26, #2
    fmlalb z28.s, z4.h, z3.h[6]
    fmlalb z29.s, z5.h, z3.h[6]
    fmlalb z30.s, z6.h, z3.h[6]
    fmlalb z31.s, z7.h, z3.h[6]
    fmlalt z28.s, z4.h, z3.h[7]
    fmlalt z29.s, z5.h, z3.h[7]
    fmlalt z30.s, z6.h, z3.h[7]
    fmlalt z31.s, z7.h, z3.h[7]
    fmlalb z8.s, z20.h, z3.h[6]
    fmlalb z9.s, z21.h, z3.h[6]
    fmlalb z10.s, z22.h, z3.h[6]
    fmlalb z11.s, z23.h, z3.h[6]
    fmlalt z8.s, z20.h, z3.h[7]
    fmlalt z9.s, z21.h, z3.h[7]
    fmlalt z10.s, z22.h, z3.h[7]
    fmlalt z11.s, z23.h, z3.h[7]
KAI_ASM_LABEL(label_25)  // Width 4: Multiply loop: multiply skip
    tbz x15, #1, label_26
    add x21, x0, #0x2
    add x20, x0, #0x0
    ld1rh { z1.h }, p1/Z, [x21]
    ld1rh { z0.h }, p1/Z, [x20]
    fcvt z12.h, p1/M, z12.s
    fcvt z7.h, p1/M, z13.s
    uzp1 z7.h, z12.h, z7.h
    fcvt z12.h, p1/M, z14.s
    fcvt z15.h, p1/M, z15.s
    uzp1 z15.h, z12.h, z15.h
    FMIN z7.h, p1/M, z7.h, z0.h
    FMAX z7.h, p1/M, z7.h, z1.h
    FMIN z15.h, p1/M, z15.h, z0.h
    FMAX z15.h, p1/M, z15.h, z1.h

    st1h z7.h, p1, [x13]
    st1h z15.h, p1, [x13, #1, MUL VL]
    addvl x13, x13, #2
    fcvt z12.h, p1/M, z16.s
    fcvt z7.h, p1/M, z17.s
    uzp1 z7.h, z12.h, z7.h
    fcvt z12.h, p1/M, z18.s
    fcvt z15.h, p1/M, z19.s
    uzp1 z15.h, z12.h, z15.h
    FMIN z7.h, p1/M, z7.h, z0.h
    FMAX z7.h, p1/M, z7.h, z1.h
    FMIN z15.h, p1/M, z15.h, z0.h
    FMAX z15.h, p1/M, z15.h, z1.h

    st1h z7.h, p1, [x13]
    st1h z15.h, p1, [x13, #1, MUL VL]
    addvl x13, x13, #2
    fcvt z12.h, p1/M, z28.s
    fcvt z14.h, p1/M, z29.s
    uzp1 z14.h, z12.h, z14.h
    fcvt z12.h, p1/M, z30.s
    fcvt z15.h, p1/M, z31.s
    uzp1 z15.h, z12.h, z15.h
    FMIN z14.h, p1/M, z14.h, z0.h
    FMAX z14.h, p1/M, z14.h, z1.h
    FMIN z15.h, p1/M, z15.h, z0.h
    FMAX z15.h, p1/M, z15.h, z1.h

    st1h z14.h, p1, [x13]
    st1h z15.h, p1, [x13, #1, MUL VL]
    addvl x13, x13, #2
    fcvt z12.h, p1/M, z8.s
    fcvt z18.h, p1/M, z9.s
    uzp1 z18.h, z12.h, z18.h
    fcvt z12.h, p1/M, z10.s
    fcvt z26.h, p1/M, z11.s
    uzp1 z26.h, z12.h, z26.h
    FMIN z18.h, p1/M, z18.h, z0.h
    FMAX z18.h, p1/M, z18.h, z1.h
    FMIN z26.h, p1/M, z26.h, z0.h
    FMAX z26.h, p1/M, z26.h, z1.h

    st1h z18.h, p2, [x13]
    st1h z26.h, p3, [x13, #1, MUL VL]
    addvl x13, x13, #2
    b label_27
KAI_ASM_LABEL(label_26)  // Width 4: No activation
    fcvt z12.h, p1/M, z12.s
    fcvt z7.h, p1/M, z13.s
    uzp1 z7.h, z12.h, z7.h
    fcvt z12.h, p1/M, z14.s
    fcvt z15.h, p1/M, z15.s
    uzp1 z15.h, z12.h, z15.h
    st1h z7.h, p1, [x13]
    st1h z15.h, p1, [x13, #1, MUL VL]
    addvl x13, x13, #2
    fcvt z12.h, p1/M, z16.s
    fcvt z7.h, p1/M, z17.s
    uzp1 z7.h, z12.h, z7.h
    fcvt z12.h, p1/M, z18.s
    fcvt z15.h, p1/M, z19.s
    uzp1 z15.h, z12.h, z15.h
    st1h z7.h, p1, [x13]
    st1h z15.h, p1, [x13, #1, MUL VL]
    addvl x13, x13, #2
    fcvt z12.h, p1/M, z28.s
    fcvt z14.h, p1/M, z29.s
    uzp1 z14.h, z12.h, z14.h
    fcvt z12.h, p1/M, z30.s
    fcvt z15.h, p1/M, z31.s
    uzp1 z15.h, z12.h, z15.h
    st1h z14.h, p1, [x13]
    st1h z15.h, p1, [x13, #1, MUL VL]
    addvl x13, x13, #2
    fcvt z12.h, p1/M, z8.s
    fcvt z18.h, p1/M, z9.s
    uzp1 z18.h, z12.h, z18.h
    fcvt z12.h, p1/M, z10.s
    fcvt z26.h, p1/M, z11.s
    uzp1 z26.h, z12.h, z26.h
    st1h z18.h, p2, [x13]
    st1h z26.h, p3, [x13, #1, MUL VL]
    addvl x13, x13, #2
KAI_ASM_LABEL(label_27)  // Width 4: Output done
    subs x17, x17, #0x4
    mov x14, x22
    sub x5, x5, x6, LSL #2
    bgt label_4
KAI_ASM_LABEL(label_28)  // Exit
    SMSTOP
    ldp x22, x23, [sp, 16]
    ldp x22, x23, [sp, 16]
    ldp x24, x25, [sp, 32]
    ldp x26, x27, [sp, 48]
    ldr x28, [sp, 64]
    ldp d8, d9, [sp, 72]
    ldp d10, d11, [sp, 88]
    ldp d12, d13, [sp, 104]
    ldp d14, d15, [sp, 120]
    ldp x20, x21, [sp], 144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme1_dot)

    KAI_ASM_END
