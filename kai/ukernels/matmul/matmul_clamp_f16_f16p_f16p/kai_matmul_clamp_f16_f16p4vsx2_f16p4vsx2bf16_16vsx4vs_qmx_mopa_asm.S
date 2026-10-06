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

    KAI_ASM_CODE(matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_16vsx4vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_16vsx4vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_16vsx4vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_16vsx4vs_qmx_mopa)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)  // SMSTART ZA
    ptrue p1.b
    mov x7, #0
    ldr x8, [x0, #0x18]
    cntw x17
    ldr x16, [x0, #0x8]
    add x8, x8, #1
    ldr x15, [x0, #0x10]
    ldr x14, [x0, #0x28]
    lsr x8, x8, #1
    ldr x13, [x0, #0x38]
    ldr x11, [x0, #0x48]
    ldr x20, [x0, #0x78]
    KAI_ASM_INST(0x84c0a68c)  // ld1rh { z12.h }, p1/Z, [x20]
    KAI_ASM_INST(0x84c1a68d)  // ld1rh { z13.h }, p1/Z, [x20, #2]
    ldr x10, [x0, #0x20]

KAI_ASM_LABEL(label_1)  // M loop
    mov x9, #0
    ldr x28, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // N loop
    mov x27, x28
    mov x26, x10
    add x25, x26, x14
    add x24, x26, x14, LSL #1
    add x23, x25, x14, LSL #1
    KAI_ASM_INST(0xc00800ff)  // zero { za }
    whilelt p0.s, x9, x15

    // FP16 bias occupies one packed vector and is widened in-place.  A
    // horizontal add seeds every active ZA tile without ZIP or grouped
    // SME2 operations.
    ld1h { z0.s }, p1/Z, [x27]
    incw x27, ALL, MUL #2
    fcvt z0.s, p1/M, z0.h
    KAI_ASM_INST(0xc0902400)  // addha za0.s, p1/M, p1/M, z0.s
    KAI_ASM_INST(0xc0902401)  // addha za1.s, p1/M, p1/M, z0.s
    KAI_ASM_INST(0xc0902402)  // addha za2.s, p1/M, p1/M, z0.s
    KAI_ASM_INST(0xc0902403)  // addha za3.s, p1/M, p1/M, z0.s

    // FP16 FMOPA widens into FP32 ZA and consumes two original K values.
    mov x22, x8
    cbz x22, label_4
KAI_ASM_LABEL(label_3)  // K loop: individual, non-grouped LD1H
    ld1h { z0.h }, p1/Z, [x26]
    ld1h { z4.h }, p1/Z, [x25]
    ld1h { z8.h }, p1/Z, [x24]
    ld1h { z16.h }, p1/Z, [x23]
    ld1h { z20.h }, p1/Z, [x27]
    addvl x26, x26, #1
    addvl x25, x25, #1
    addvl x24, x24, #1
    addvl x23, x23, #1
    addvl x27, x27, #1
    KAI_ASM_INST(0x81b42400)  // fmopa za0.s, p1/M, p1/M, z0.h, z20.h
    KAI_ASM_INST(0x81b42481)  // fmopa za1.s, p1/M, p1/M, z4.h, z20.h
    KAI_ASM_INST(0x81b42502)  // fmopa za2.s, p1/M, p1/M, z8.h, z20.h
    KAI_ASM_INST(0x81b42603)  // fmopa za3.s, p1/M, p1/M, z16.h, z20.h
    subs x22, x22, #1
    bne label_3

KAI_ASM_LABEL(label_4)  // Output
    // Preserve the next packed-RHS panel before reusing x28 for C.
    add x6, x28, x13
    ldr x28, [x0, #0x40]
    madd x28, x11, x7, x28
    add x28, x28, x9, LSL #1
    madd x27, x17, x11, x28
    madd x26, x17, x11, x27
    madd x25, x17, x11, x26
    mov x12, #0

    // Extract each FP32 ZA row separately.  Conversion, FP16 clamp and
    // store are deliberately individual base-SME/SVE operations.
    sub x21, x16, x7
    cmp x21, x17
    csel x21, x21, x17, LS
KAI_ASM_LABEL(label_5)  // Tile 0
    KAI_ASM_INST(0xc0820400)  // mova z0.s, p1/M, za0h.s[w12, 0]
    add x12, x12, #1
    fcvt z0.h, p1/M, z0.s
    fmin z0.h, p1/M, z0.h, z13.h
    fmax z0.h, p1/M, z0.h, z12.h
    st1h { z0.s }, p0, [x28]
    add x28, x28, x11
    subs x21, x21, #1
    bne label_5

    add x1, x7, x17
    cmp x16, x1
    bls label_9
    sub x21, x16, x1
    cmp x21, x17
    csel x21, x21, x17, LS
KAI_ASM_LABEL(label_6)  // Tile 1
    KAI_ASM_INST(0xc0820480)  // mova z0.s, p1/M, za1h.s[w12, 0]
    add x12, x12, #1
    fcvt z0.h, p1/M, z0.s
    fmin z0.h, p1/M, z0.h, z13.h
    fmax z0.h, p1/M, z0.h, z12.h
    st1h { z0.s }, p0, [x27]
    add x27, x27, x11
    subs x21, x21, #1
    bne label_6

    add x1, x7, x17, LSL #1
    cmp x16, x1
    bls label_9
    sub x21, x16, x1
    cmp x21, x17
    csel x21, x21, x17, LS
KAI_ASM_LABEL(label_7)  // Tile 2
    KAI_ASM_INST(0xc0820500)  // mova z0.s, p1/M, za2h.s[w12, 0]
    add x12, x12, #1
    fcvt z0.h, p1/M, z0.s
    fmin z0.h, p1/M, z0.h, z13.h
    fmax z0.h, p1/M, z0.h, z12.h
    st1h { z0.s }, p0, [x26]
    add x26, x26, x11
    subs x21, x21, #1
    bne label_7

    add x1, x7, x17, LSL #1
    add x1, x1, x17
    cmp x16, x1
    bls label_9
    sub x21, x16, x1
    cmp x21, x17
    csel x21, x21, x17, LS
KAI_ASM_LABEL(label_8)  // Tile 3
    KAI_ASM_INST(0xc0820580)  // mova z0.s, p1/M, za3h.s[w12, 0]
    add x12, x12, #1
    fcvt z0.h, p1/M, z0.s
    fmin z0.h, p1/M, z0.h, z13.h
    fmax z0.h, p1/M, z0.h, z12.h
    st1h { z0.s }, p0, [x25]
    add x25, x25, x11
    subs x21, x21, #1
    bne label_8

KAI_ASM_LABEL(label_9)
    incw x9
    cmp x9, x15
    bge label_10
    mov x28, x6
    b label_2
KAI_ASM_LABEL(label_10)
    incw x7, ALL, MUL #4
    add x10, x10, x14, LSL #2
    cmp x7, x16
    blt label_1

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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_16vsx4vs_qmx_mopa)

    KAI_ASM_END
