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
    #define KAI_ASM_ALIGN_LOOP
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
    #define KAI_ASM_ALIGN_LOOP .p2align 6
    #define KAI_ASM_LABEL(name) name:
    #define KAI_ASM_INST(hex) .inst hex
    #define KAI_ASM_END
#endif

    KAI_ASM_CODE(matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_qmx_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_qmx_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_qmx_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_qmx_dot)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART
    ldr x6, [x0, #0x18]
    cntw x7, ALL, MUL #4
    ptrue p1.b
    ldr x17, [x0, #0x20]
    ldr x13, [x0, #0x10]
    add x16, x6, x7
    ldr x12, [x0, #0x28]
    sub x16, x16, #0x1
    ldr x15, [x0, #0x8]
    udiv x16, x16, x7
    ldr x14, [x0, #0x30]

    // A packed RHS panel contains one FP16 bias vector followed by one FP16
    // vector for each pair of K values.
    add x11, x17, #0x1
    cntw x20
    bic x11, x11, #0x1
    lsl x11, x11, #0x1
    add x11, x11, #0x2
    mul x11, x11, x20

KAI_ASM_LABEL(label_column_loop)
    cmp x16, #0x2
    bge label_width_2

    // Width 1: four accumulator vectors, covering four packed RHS panels.
    cntw x20
    add x24, x13, x11
    ld1h { z8.s }, p1/Z, [x13]
    cmp x6, x20
    cntw x20, ALL, MUL #2
    add x23, x13, x11, LSL #1
    csel x24, x24, x13, GT
    cmp x6, x20
    cntw x20, ALL, MUL #3
    add x22, x23, x11
    ld1h { z9.s }, p1/Z, [x24]
    csel x23, x23, x13, GT
    cmp x6, x20
    ld1h { z10.s }, p1/Z, [x23]
    csel x22, x22, x13, GT
    ld1h { z11.s }, p1/Z, [x22]
    fcvt z8.s, p1/m, z8.h
    fcvt z9.s, p1/m, z9.h
    fcvt z10.s, p1/m, z10.h
    fcvt z11.s, p1/m, z11.h

    mov x10, x17
    mov x9, x15
    mov x21, x6
    mov x8, xzr
    whilelt p4.h, x8, x21
    inch x8
    whilelt p5.h, x8, x21

    inch x13
    inch x24
    inch x23
    inch x22
    lsl x20, x17, #0x1
    KAI_ASM_INST(0xf8b44938)  // rprfm pldkeep, x20, [x9]
    cmp x10, #0x8
    ble label_width_1_k_tail

    KAI_ASM_ALIGN_LOOP
KAI_ASM_LABEL(label_width_1_k_main)
    whilelt p0.h, XZR, x10
    ld1rqh { z3.h }, p0/Z, [x9]
    sub x10, x10, #0x8
    add x9, x9, #0x10

    ldnt1h { z16.h }, p1/Z, [x13]
    ldnt1h { z17.h }, p1/Z, [x24]
    ldnt1h { z18.h }, p1/Z, [x23]
    ldnt1h { z19.h }, p1/Z, [x22]
    KAI_ASM_INST(0x64a34208)  // fmlalb z8.s, z16.h, z3.h[0]
    KAI_ASM_INST(0x64a34229)  // fmlalb z9.s, z17.h, z3.h[0]
    KAI_ASM_INST(0x64a3424a)  // fmlalb z10.s, z18.h, z3.h[0]
    KAI_ASM_INST(0x64a3426b)  // fmlalb z11.s, z19.h, z3.h[0]
    KAI_ASM_INST(0x64a34e08)  // fmlalt z8.s, z16.h, z3.h[1]
    KAI_ASM_INST(0x64a34e29)  // fmlalt z9.s, z17.h, z3.h[1]
    KAI_ASM_INST(0x64a34e4a)  // fmlalt z10.s, z18.h, z3.h[1]
    KAI_ASM_INST(0x64a34e6b)  // fmlalt z11.s, z19.h, z3.h[1]

    ldnt1h { z16.h }, p1/Z, [x13, #1, MUL VL]
    ldnt1h { z17.h }, p1/Z, [x24, #1, MUL VL]
    ldnt1h { z18.h }, p1/Z, [x23, #1, MUL VL]
    ldnt1h { z19.h }, p1/Z, [x22, #1, MUL VL]
    KAI_ASM_INST(0x64ab4208)  // fmlalb z8.s, z16.h, z3.h[2]
    KAI_ASM_INST(0x64ab4229)  // fmlalb z9.s, z17.h, z3.h[2]
    KAI_ASM_INST(0x64ab424a)  // fmlalb z10.s, z18.h, z3.h[2]
    KAI_ASM_INST(0x64ab426b)  // fmlalb z11.s, z19.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e08)  // fmlalt z8.s, z16.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e29)  // fmlalt z9.s, z17.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e4a)  // fmlalt z10.s, z18.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e6b)  // fmlalt z11.s, z19.h, z3.h[3]

    ldnt1h { z16.h }, p1/Z, [x13, #2, MUL VL]
    ldnt1h { z17.h }, p1/Z, [x24, #2, MUL VL]
    ldnt1h { z18.h }, p1/Z, [x23, #2, MUL VL]
    ldnt1h { z19.h }, p1/Z, [x22, #2, MUL VL]
    KAI_ASM_INST(0x64b34208)  // fmlalb z8.s, z16.h, z3.h[4]
    KAI_ASM_INST(0x64b34229)  // fmlalb z9.s, z17.h, z3.h[4]
    KAI_ASM_INST(0x64b3424a)  // fmlalb z10.s, z18.h, z3.h[4]
    KAI_ASM_INST(0x64b3426b)  // fmlalb z11.s, z19.h, z3.h[4]
    KAI_ASM_INST(0x64b34e08)  // fmlalt z8.s, z16.h, z3.h[5]
    KAI_ASM_INST(0x64b34e29)  // fmlalt z9.s, z17.h, z3.h[5]
    KAI_ASM_INST(0x64b34e4a)  // fmlalt z10.s, z18.h, z3.h[5]
    KAI_ASM_INST(0x64b34e6b)  // fmlalt z11.s, z19.h, z3.h[5]

    ldnt1h { z16.h }, p1/Z, [x13, #3, MUL VL]
    ldnt1h { z17.h }, p1/Z, [x24, #3, MUL VL]
    ldnt1h { z18.h }, p1/Z, [x23, #3, MUL VL]
    ldnt1h { z19.h }, p1/Z, [x22, #3, MUL VL]
    KAI_ASM_INST(0x64bb4208)  // fmlalb z8.s, z16.h, z3.h[6]
    KAI_ASM_INST(0x64bb4229)  // fmlalb z9.s, z17.h, z3.h[6]
    KAI_ASM_INST(0x64bb424a)  // fmlalb z10.s, z18.h, z3.h[6]
    KAI_ASM_INST(0x64bb426b)  // fmlalb z11.s, z19.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e08)  // fmlalt z8.s, z16.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e29)  // fmlalt z9.s, z17.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e4a)  // fmlalt z10.s, z18.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e6b)  // fmlalt z11.s, z19.h, z3.h[7]

    addvl x13, x13, #4
    addvl x24, x24, #4
    addvl x23, x23, #4
    addvl x22, x22, #4
    cmp x10, #0x8
    bgt label_width_1_k_main

KAI_ASM_LABEL(label_width_1_k_tail)
    whilelt p0.h, XZR, x10
    ld1rqh { z3.h }, p0/Z, [x9]
    subs x10, x10, #0x2
    ldnt1h { z16.h }, p1/Z, [x13]
    addvl x13, x13, #1
    ldnt1h { z17.h }, p1/Z, [x24]
    addvl x24, x24, #1
    ldnt1h { z18.h }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1h { z19.h }, p1/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0x64a34208)  // fmlalb z8.s, z16.h, z3.h[0]
    KAI_ASM_INST(0x64a34229)  // fmlalb z9.s, z17.h, z3.h[0]
    KAI_ASM_INST(0x64a3424a)  // fmlalb z10.s, z18.h, z3.h[0]
    KAI_ASM_INST(0x64a3426b)  // fmlalb z11.s, z19.h, z3.h[0]
    KAI_ASM_INST(0x64a34e08)  // fmlalt z8.s, z16.h, z3.h[1]
    KAI_ASM_INST(0x64a34e29)  // fmlalt z9.s, z17.h, z3.h[1]
    KAI_ASM_INST(0x64a34e4a)  // fmlalt z10.s, z18.h, z3.h[1]
    KAI_ASM_INST(0x64a34e6b)  // fmlalt z11.s, z19.h, z3.h[1]
    ble label_width_1_k_done

    subs x10, x10, #0x2
    ldnt1h { z16.h }, p1/Z, [x13]
    addvl x13, x13, #1
    ldnt1h { z17.h }, p1/Z, [x24]
    addvl x24, x24, #1
    ldnt1h { z18.h }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1h { z19.h }, p1/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0x64ab4208)  // fmlalb z8.s, z16.h, z3.h[2]
    KAI_ASM_INST(0x64ab4229)  // fmlalb z9.s, z17.h, z3.h[2]
    KAI_ASM_INST(0x64ab424a)  // fmlalb z10.s, z18.h, z3.h[2]
    KAI_ASM_INST(0x64ab426b)  // fmlalb z11.s, z19.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e08)  // fmlalt z8.s, z16.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e29)  // fmlalt z9.s, z17.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e4a)  // fmlalt z10.s, z18.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e6b)  // fmlalt z11.s, z19.h, z3.h[3]
    ble label_width_1_k_done

    subs x10, x10, #0x2
    ldnt1h { z16.h }, p1/Z, [x13]
    addvl x13, x13, #1
    ldnt1h { z17.h }, p1/Z, [x24]
    addvl x24, x24, #1
    ldnt1h { z18.h }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1h { z19.h }, p1/Z, [x22]
    addvl x22, x22, #1
    KAI_ASM_INST(0x64b34208)  // fmlalb z8.s, z16.h, z3.h[4]
    KAI_ASM_INST(0x64b34229)  // fmlalb z9.s, z17.h, z3.h[4]
    KAI_ASM_INST(0x64b3424a)  // fmlalb z10.s, z18.h, z3.h[4]
    KAI_ASM_INST(0x64b3426b)  // fmlalb z11.s, z19.h, z3.h[4]
    KAI_ASM_INST(0x64b34e08)  // fmlalt z8.s, z16.h, z3.h[5]
    KAI_ASM_INST(0x64b34e29)  // fmlalt z9.s, z17.h, z3.h[5]
    KAI_ASM_INST(0x64b34e4a)  // fmlalt z10.s, z18.h, z3.h[5]
    KAI_ASM_INST(0x64b34e6b)  // fmlalt z11.s, z19.h, z3.h[5]
    ble label_width_1_k_done

    ldnt1h { z16.h }, p1/Z, [x13]
    ldnt1h { z17.h }, p1/Z, [x24]
    ldnt1h { z18.h }, p1/Z, [x23]
    ldnt1h { z19.h }, p1/Z, [x22]
    KAI_ASM_INST(0x64bb4208)  // fmlalb z8.s, z16.h, z3.h[6]
    KAI_ASM_INST(0x64bb4229)  // fmlalb z9.s, z17.h, z3.h[6]
    KAI_ASM_INST(0x64bb424a)  // fmlalb z10.s, z18.h, z3.h[6]
    KAI_ASM_INST(0x64bb426b)  // fmlalb z11.s, z19.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e08)  // fmlalt z8.s, z16.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e29)  // fmlalt z9.s, z17.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e4a)  // fmlalt z10.s, z18.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e6b)  // fmlalt z11.s, z19.h, z3.h[7]

KAI_ASM_LABEL(label_width_1_k_done)
    fcvt z16.h, p1/m, z8.s
    fcvt z17.h, p1/m, z9.s
    uzp1 z20.h, z16.h, z17.h
    fcvt z18.h, p1/m, z10.s
    fcvt z19.h, p1/m, z11.s
    uzp1 z21.h, z18.h, z19.h
    tbz x14, #1, label_width_1_store
    ld1rh { z22.h }, p1/Z, [x0]
    ld1rh { z23.h }, p1/Z, [x0, #2]
    fmin z20.h, p1/m, z20.h, z22.h
    fmax z20.h, p1/m, z20.h, z23.h
    fmin z21.h, p1/m, z21.h, z22.h
    fmax z21.h, p1/m, z21.h, z23.h
KAI_ASM_LABEL(label_width_1_store)
    st1h { z20.h }, p4, [x12]
    st1h { z21.h }, p5, [x12, #1, MUL VL]
    b label_exit

KAI_ASM_LABEL(label_width_2)
    // Width 2: eight accumulator vectors, covering eight packed RHS panels.
    add x28, x13, x11, LSL #2
    cntw x20, ALL, MUL #5
    ld1h { z8.s }, p1/Z, [x13]
    add x27, x28, x11
    cmp x6, x20
    ld1h { z12.s }, p1/Z, [x28]
    cntw x20, ALL, MUL #6
    add x26, x28, x11, LSL #1
    csel x27, x27, x13, GT
    cmp x6, x20
    cntw x20, ALL, MUL #7
    add x25, x13, x11, LSL #1
    ld1h { z13.s }, p1/Z, [x27]
    add x24, x26, x11
    csel x26, x26, x13, GT
    ld1h { z10.s }, p1/Z, [x25]
    cmp x6, x20
    add x23, x13, x11
    ld1h { z14.s }, p1/Z, [x26]
    add x22, x25, x11
    csel x24, x24, x13, GT
    ld1h { z9.s }, p1/Z, [x23]
    ld1h { z11.s }, p1/Z, [x22]
    ld1h { z15.s }, p1/Z, [x24]
    fcvt z8.s, p1/m, z8.h
    fcvt z9.s, p1/m, z9.h
    fcvt z10.s, p1/m, z10.h
    fcvt z11.s, p1/m, z11.h
    fcvt z12.s, p1/m, z12.h
    fcvt z13.s, p1/m, z13.h
    fcvt z14.s, p1/m, z14.h
    fcvt z15.s, p1/m, z15.h

    mov x10, x17
    mov x9, x15
    sub x21, x6, x7
    mov x8, xzr
    whilelt p4.h, x8, x21
    inch x8
    whilelt p5.h, x8, x21

    lsl x20, x17, #0x1
    KAI_ASM_INST(0xf8b44938)  // rprfm pldkeep, x20, [x9]
    add x20, x13, x11, LSL #3
    inch x13
    inch x23
    inch x25
    inch x22
    inch x28
    inch x27
    inch x26
    inch x24
    cmp x10, #0x8
    ble label_width_2_k_tail

    KAI_ASM_ALIGN_LOOP
KAI_ASM_LABEL(label_width_2_k_main)
    whilelt p0.h, XZR, x10
    ld1rqh { z3.h }, p0/Z, [x9]
    sub x10, x10, #0x8
    add x9, x9, #0x10

    ldnt1h { z16.h }, p1/Z, [x13]
    ldnt1h { z17.h }, p1/Z, [x23]
    ldnt1h { z18.h }, p1/Z, [x25]
    ldnt1h { z19.h }, p1/Z, [x22]
    ldnt1h { z20.h }, p1/Z, [x28]
    ldnt1h { z21.h }, p1/Z, [x27]
    ldnt1h { z22.h }, p1/Z, [x26]
    ldnt1h { z23.h }, p1/Z, [x24]
    KAI_ASM_INST(0x64a34208)  // fmlalb z8.s, z16.h, z3.h[0]
    KAI_ASM_INST(0x64a34229)  // fmlalb z9.s, z17.h, z3.h[0]
    KAI_ASM_INST(0x64a3424a)  // fmlalb z10.s, z18.h, z3.h[0]
    KAI_ASM_INST(0x64a3426b)  // fmlalb z11.s, z19.h, z3.h[0]
    KAI_ASM_INST(0x64a3428c)  // fmlalb z12.s, z20.h, z3.h[0]
    KAI_ASM_INST(0x64a342ad)  // fmlalb z13.s, z21.h, z3.h[0]
    KAI_ASM_INST(0x64a342ce)  // fmlalb z14.s, z22.h, z3.h[0]
    KAI_ASM_INST(0x64a342ef)  // fmlalb z15.s, z23.h, z3.h[0]
    KAI_ASM_INST(0x64a34e08)  // fmlalt z8.s, z16.h, z3.h[1]
    KAI_ASM_INST(0x64a34e29)  // fmlalt z9.s, z17.h, z3.h[1]
    KAI_ASM_INST(0x64a34e4a)  // fmlalt z10.s, z18.h, z3.h[1]
    KAI_ASM_INST(0x64a34e6b)  // fmlalt z11.s, z19.h, z3.h[1]
    KAI_ASM_INST(0x64a34e8c)  // fmlalt z12.s, z20.h, z3.h[1]
    KAI_ASM_INST(0x64a34ead)  // fmlalt z13.s, z21.h, z3.h[1]
    KAI_ASM_INST(0x64a34ece)  // fmlalt z14.s, z22.h, z3.h[1]
    KAI_ASM_INST(0x64a34eef)  // fmlalt z15.s, z23.h, z3.h[1]

    ldnt1h { z16.h }, p1/Z, [x13, #1, MUL VL]
    ldnt1h { z17.h }, p1/Z, [x23, #1, MUL VL]
    ldnt1h { z18.h }, p1/Z, [x25, #1, MUL VL]
    ldnt1h { z19.h }, p1/Z, [x22, #1, MUL VL]
    ldnt1h { z20.h }, p1/Z, [x28, #1, MUL VL]
    ldnt1h { z21.h }, p1/Z, [x27, #1, MUL VL]
    ldnt1h { z22.h }, p1/Z, [x26, #1, MUL VL]
    ldnt1h { z23.h }, p1/Z, [x24, #1, MUL VL]
    KAI_ASM_INST(0x64ab4208)  // fmlalb z8.s, z16.h, z3.h[2]
    KAI_ASM_INST(0x64ab4229)  // fmlalb z9.s, z17.h, z3.h[2]
    KAI_ASM_INST(0x64ab424a)  // fmlalb z10.s, z18.h, z3.h[2]
    KAI_ASM_INST(0x64ab426b)  // fmlalb z11.s, z19.h, z3.h[2]
    KAI_ASM_INST(0x64ab428c)  // fmlalb z12.s, z20.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ad)  // fmlalb z13.s, z21.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ce)  // fmlalb z14.s, z22.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ef)  // fmlalb z15.s, z23.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e08)  // fmlalt z8.s, z16.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e29)  // fmlalt z9.s, z17.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e4a)  // fmlalt z10.s, z18.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e6b)  // fmlalt z11.s, z19.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e8c)  // fmlalt z12.s, z20.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ead)  // fmlalt z13.s, z21.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ece)  // fmlalt z14.s, z22.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eef)  // fmlalt z15.s, z23.h, z3.h[3]

    ldnt1h { z16.h }, p1/Z, [x13, #2, MUL VL]
    ldnt1h { z17.h }, p1/Z, [x23, #2, MUL VL]
    ldnt1h { z18.h }, p1/Z, [x25, #2, MUL VL]
    ldnt1h { z19.h }, p1/Z, [x22, #2, MUL VL]
    ldnt1h { z20.h }, p1/Z, [x28, #2, MUL VL]
    ldnt1h { z21.h }, p1/Z, [x27, #2, MUL VL]
    ldnt1h { z22.h }, p1/Z, [x26, #2, MUL VL]
    ldnt1h { z23.h }, p1/Z, [x24, #2, MUL VL]
    KAI_ASM_INST(0x64b34208)  // fmlalb z8.s, z16.h, z3.h[4]
    KAI_ASM_INST(0x64b34229)  // fmlalb z9.s, z17.h, z3.h[4]
    KAI_ASM_INST(0x64b3424a)  // fmlalb z10.s, z18.h, z3.h[4]
    KAI_ASM_INST(0x64b3426b)  // fmlalb z11.s, z19.h, z3.h[4]
    KAI_ASM_INST(0x64b3428c)  // fmlalb z12.s, z20.h, z3.h[4]
    KAI_ASM_INST(0x64b342ad)  // fmlalb z13.s, z21.h, z3.h[4]
    KAI_ASM_INST(0x64b342ce)  // fmlalb z14.s, z22.h, z3.h[4]
    KAI_ASM_INST(0x64b342ef)  // fmlalb z15.s, z23.h, z3.h[4]
    KAI_ASM_INST(0x64b34e08)  // fmlalt z8.s, z16.h, z3.h[5]
    KAI_ASM_INST(0x64b34e29)  // fmlalt z9.s, z17.h, z3.h[5]
    KAI_ASM_INST(0x64b34e4a)  // fmlalt z10.s, z18.h, z3.h[5]
    KAI_ASM_INST(0x64b34e6b)  // fmlalt z11.s, z19.h, z3.h[5]
    KAI_ASM_INST(0x64b34e8c)  // fmlalt z12.s, z20.h, z3.h[5]
    KAI_ASM_INST(0x64b34ead)  // fmlalt z13.s, z21.h, z3.h[5]
    KAI_ASM_INST(0x64b34ece)  // fmlalt z14.s, z22.h, z3.h[5]
    KAI_ASM_INST(0x64b34eef)  // fmlalt z15.s, z23.h, z3.h[5]

    ldnt1h { z16.h }, p1/Z, [x13, #3, MUL VL]
    ldnt1h { z17.h }, p1/Z, [x23, #3, MUL VL]
    ldnt1h { z18.h }, p1/Z, [x25, #3, MUL VL]
    ldnt1h { z19.h }, p1/Z, [x22, #3, MUL VL]
    ldnt1h { z20.h }, p1/Z, [x28, #3, MUL VL]
    ldnt1h { z21.h }, p1/Z, [x27, #3, MUL VL]
    ldnt1h { z22.h }, p1/Z, [x26, #3, MUL VL]
    ldnt1h { z23.h }, p1/Z, [x24, #3, MUL VL]
    KAI_ASM_INST(0x64bb4208)  // fmlalb z8.s, z16.h, z3.h[6]
    KAI_ASM_INST(0x64bb4229)  // fmlalb z9.s, z17.h, z3.h[6]
    KAI_ASM_INST(0x64bb424a)  // fmlalb z10.s, z18.h, z3.h[6]
    KAI_ASM_INST(0x64bb426b)  // fmlalb z11.s, z19.h, z3.h[6]
    KAI_ASM_INST(0x64bb428c)  // fmlalb z12.s, z20.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ad)  // fmlalb z13.s, z21.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ce)  // fmlalb z14.s, z22.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ef)  // fmlalb z15.s, z23.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e08)  // fmlalt z8.s, z16.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e29)  // fmlalt z9.s, z17.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e4a)  // fmlalt z10.s, z18.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e6b)  // fmlalt z11.s, z19.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e8c)  // fmlalt z12.s, z20.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ead)  // fmlalt z13.s, z21.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ece)  // fmlalt z14.s, z22.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eef)  // fmlalt z15.s, z23.h, z3.h[7]

    addvl x13, x13, #4
    addvl x23, x23, #4
    addvl x25, x25, #4
    addvl x22, x22, #4
    addvl x28, x28, #4
    addvl x27, x27, #4
    addvl x26, x26, #4
    addvl x24, x24, #4
    cmp x10, #0x8
    bgt label_width_2_k_main

KAI_ASM_LABEL(label_width_2_k_tail)
    whilelt p0.h, XZR, x10
    ld1rqh { z3.h }, p0/Z, [x9]
    subs x10, x10, #0x2
    ldnt1h { z16.h }, p1/Z, [x13]
    addvl x13, x13, #1
    ldnt1h { z17.h }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1h { z18.h }, p1/Z, [x25]
    addvl x25, x25, #1
    ldnt1h { z19.h }, p1/Z, [x22]
    addvl x22, x22, #1
    ldnt1h { z20.h }, p1/Z, [x28]
    addvl x28, x28, #1
    ldnt1h { z21.h }, p1/Z, [x27]
    addvl x27, x27, #1
    ldnt1h { z22.h }, p1/Z, [x26]
    addvl x26, x26, #1
    ldnt1h { z23.h }, p1/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0x64a34208)  // fmlalb z8.s, z16.h, z3.h[0]
    KAI_ASM_INST(0x64a34229)  // fmlalb z9.s, z17.h, z3.h[0]
    KAI_ASM_INST(0x64a3424a)  // fmlalb z10.s, z18.h, z3.h[0]
    KAI_ASM_INST(0x64a3426b)  // fmlalb z11.s, z19.h, z3.h[0]
    KAI_ASM_INST(0x64a3428c)  // fmlalb z12.s, z20.h, z3.h[0]
    KAI_ASM_INST(0x64a342ad)  // fmlalb z13.s, z21.h, z3.h[0]
    KAI_ASM_INST(0x64a342ce)  // fmlalb z14.s, z22.h, z3.h[0]
    KAI_ASM_INST(0x64a342ef)  // fmlalb z15.s, z23.h, z3.h[0]
    KAI_ASM_INST(0x64a34e08)  // fmlalt z8.s, z16.h, z3.h[1]
    KAI_ASM_INST(0x64a34e29)  // fmlalt z9.s, z17.h, z3.h[1]
    KAI_ASM_INST(0x64a34e4a)  // fmlalt z10.s, z18.h, z3.h[1]
    KAI_ASM_INST(0x64a34e6b)  // fmlalt z11.s, z19.h, z3.h[1]
    KAI_ASM_INST(0x64a34e8c)  // fmlalt z12.s, z20.h, z3.h[1]
    KAI_ASM_INST(0x64a34ead)  // fmlalt z13.s, z21.h, z3.h[1]
    KAI_ASM_INST(0x64a34ece)  // fmlalt z14.s, z22.h, z3.h[1]
    KAI_ASM_INST(0x64a34eef)  // fmlalt z15.s, z23.h, z3.h[1]
    ble label_width_2_k_done

    subs x10, x10, #0x2
    ldnt1h { z16.h }, p1/Z, [x13]
    addvl x13, x13, #1
    ldnt1h { z17.h }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1h { z18.h }, p1/Z, [x25]
    addvl x25, x25, #1
    ldnt1h { z19.h }, p1/Z, [x22]
    addvl x22, x22, #1
    ldnt1h { z20.h }, p1/Z, [x28]
    addvl x28, x28, #1
    ldnt1h { z21.h }, p1/Z, [x27]
    addvl x27, x27, #1
    ldnt1h { z22.h }, p1/Z, [x26]
    addvl x26, x26, #1
    ldnt1h { z23.h }, p1/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0x64ab4208)  // fmlalb z8.s, z16.h, z3.h[2]
    KAI_ASM_INST(0x64ab4229)  // fmlalb z9.s, z17.h, z3.h[2]
    KAI_ASM_INST(0x64ab424a)  // fmlalb z10.s, z18.h, z3.h[2]
    KAI_ASM_INST(0x64ab426b)  // fmlalb z11.s, z19.h, z3.h[2]
    KAI_ASM_INST(0x64ab428c)  // fmlalb z12.s, z20.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ad)  // fmlalb z13.s, z21.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ce)  // fmlalb z14.s, z22.h, z3.h[2]
    KAI_ASM_INST(0x64ab42ef)  // fmlalb z15.s, z23.h, z3.h[2]
    KAI_ASM_INST(0x64ab4e08)  // fmlalt z8.s, z16.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e29)  // fmlalt z9.s, z17.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e4a)  // fmlalt z10.s, z18.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e6b)  // fmlalt z11.s, z19.h, z3.h[3]
    KAI_ASM_INST(0x64ab4e8c)  // fmlalt z12.s, z20.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ead)  // fmlalt z13.s, z21.h, z3.h[3]
    KAI_ASM_INST(0x64ab4ece)  // fmlalt z14.s, z22.h, z3.h[3]
    KAI_ASM_INST(0x64ab4eef)  // fmlalt z15.s, z23.h, z3.h[3]
    ble label_width_2_k_done

    subs x10, x10, #0x2
    ldnt1h { z16.h }, p1/Z, [x13]
    addvl x13, x13, #1
    ldnt1h { z17.h }, p1/Z, [x23]
    addvl x23, x23, #1
    ldnt1h { z18.h }, p1/Z, [x25]
    addvl x25, x25, #1
    ldnt1h { z19.h }, p1/Z, [x22]
    addvl x22, x22, #1
    ldnt1h { z20.h }, p1/Z, [x28]
    addvl x28, x28, #1
    ldnt1h { z21.h }, p1/Z, [x27]
    addvl x27, x27, #1
    ldnt1h { z22.h }, p1/Z, [x26]
    addvl x26, x26, #1
    ldnt1h { z23.h }, p1/Z, [x24]
    addvl x24, x24, #1
    KAI_ASM_INST(0x64b34208)  // fmlalb z8.s, z16.h, z3.h[4]
    KAI_ASM_INST(0x64b34229)  // fmlalb z9.s, z17.h, z3.h[4]
    KAI_ASM_INST(0x64b3424a)  // fmlalb z10.s, z18.h, z3.h[4]
    KAI_ASM_INST(0x64b3426b)  // fmlalb z11.s, z19.h, z3.h[4]
    KAI_ASM_INST(0x64b3428c)  // fmlalb z12.s, z20.h, z3.h[4]
    KAI_ASM_INST(0x64b342ad)  // fmlalb z13.s, z21.h, z3.h[4]
    KAI_ASM_INST(0x64b342ce)  // fmlalb z14.s, z22.h, z3.h[4]
    KAI_ASM_INST(0x64b342ef)  // fmlalb z15.s, z23.h, z3.h[4]
    KAI_ASM_INST(0x64b34e08)  // fmlalt z8.s, z16.h, z3.h[5]
    KAI_ASM_INST(0x64b34e29)  // fmlalt z9.s, z17.h, z3.h[5]
    KAI_ASM_INST(0x64b34e4a)  // fmlalt z10.s, z18.h, z3.h[5]
    KAI_ASM_INST(0x64b34e6b)  // fmlalt z11.s, z19.h, z3.h[5]
    KAI_ASM_INST(0x64b34e8c)  // fmlalt z12.s, z20.h, z3.h[5]
    KAI_ASM_INST(0x64b34ead)  // fmlalt z13.s, z21.h, z3.h[5]
    KAI_ASM_INST(0x64b34ece)  // fmlalt z14.s, z22.h, z3.h[5]
    KAI_ASM_INST(0x64b34eef)  // fmlalt z15.s, z23.h, z3.h[5]
    ble label_width_2_k_done

    ldnt1h { z16.h }, p1/Z, [x13]
    ldnt1h { z17.h }, p1/Z, [x23]
    ldnt1h { z18.h }, p1/Z, [x25]
    ldnt1h { z19.h }, p1/Z, [x22]
    ldnt1h { z20.h }, p1/Z, [x28]
    ldnt1h { z21.h }, p1/Z, [x27]
    ldnt1h { z22.h }, p1/Z, [x26]
    ldnt1h { z23.h }, p1/Z, [x24]
    KAI_ASM_INST(0x64bb4208)  // fmlalb z8.s, z16.h, z3.h[6]
    KAI_ASM_INST(0x64bb4229)  // fmlalb z9.s, z17.h, z3.h[6]
    KAI_ASM_INST(0x64bb424a)  // fmlalb z10.s, z18.h, z3.h[6]
    KAI_ASM_INST(0x64bb426b)  // fmlalb z11.s, z19.h, z3.h[6]
    KAI_ASM_INST(0x64bb428c)  // fmlalb z12.s, z20.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ad)  // fmlalb z13.s, z21.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ce)  // fmlalb z14.s, z22.h, z3.h[6]
    KAI_ASM_INST(0x64bb42ef)  // fmlalb z15.s, z23.h, z3.h[6]
    KAI_ASM_INST(0x64bb4e08)  // fmlalt z8.s, z16.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e29)  // fmlalt z9.s, z17.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e4a)  // fmlalt z10.s, z18.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e6b)  // fmlalt z11.s, z19.h, z3.h[7]
    KAI_ASM_INST(0x64bb4e8c)  // fmlalt z12.s, z20.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ead)  // fmlalt z13.s, z21.h, z3.h[7]
    KAI_ASM_INST(0x64bb4ece)  // fmlalt z14.s, z22.h, z3.h[7]
    KAI_ASM_INST(0x64bb4eef)  // fmlalt z15.s, z23.h, z3.h[7]

KAI_ASM_LABEL(label_width_2_k_done)
    // Pair the two FP32 vectors that make up each FP16 output vector.
    fcvt z16.h, p1/m, z8.s
    fcvt z17.h, p1/m, z9.s
    uzp1 z24.h, z16.h, z17.h
    fcvt z18.h, p1/m, z10.s
    fcvt z19.h, p1/m, z11.s
    uzp1 z25.h, z18.h, z19.h
    fcvt z20.h, p1/m, z12.s
    fcvt z21.h, p1/m, z13.s
    uzp1 z26.h, z20.h, z21.h
    fcvt z22.h, p1/m, z14.s
    fcvt z23.h, p1/m, z15.s
    uzp1 z27.h, z22.h, z23.h
    tbz x14, #1, label_width_2_store
    ld1rh { z28.h }, p1/Z, [x0]
    ld1rh { z29.h }, p1/Z, [x0, #2]
    fmin z24.h, p1/m, z24.h, z28.h
    fmax z24.h, p1/m, z24.h, z29.h
    fmin z25.h, p1/m, z25.h, z28.h
    fmax z25.h, p1/m, z25.h, z29.h
    fmin z26.h, p1/m, z26.h, z28.h
    fmax z26.h, p1/m, z26.h, z29.h
    fmin z27.h, p1/m, z27.h, z28.h
    fmax z27.h, p1/m, z27.h, z29.h
KAI_ASM_LABEL(label_width_2_store)
    st1h { z24.h }, p1, [x12]
    st1h { z25.h }, p1, [x12, #1, MUL VL]
    addvl x12, x12, #2
    st1h { z26.h }, p4, [x12]
    st1h { z27.h }, p5, [x12, #1, MUL VL]
    addvl x12, x12, #2

    subs x16, x16, #0x2
    mov x13, x20
    sub x6, x6, x7, LSL #1
    bgt label_column_loop

KAI_ASM_LABEL(label_exit)
    KAI_ASM_INST(0xd503467f)  // SMSTOP
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16_f16p4vsx2bf16_1x32vs_qmx_dot)

    KAI_ASM_END
