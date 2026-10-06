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

    KAI_ASM_CODE(matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx8vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx8vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx8vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx8vs_qmx_mopa)
    stp x20, x21, [sp, -144]!
    stp x22, x23, [sp, 16]
    stp x24, x25, [sp, 32]
    stp x26, x27, [sp, 48]
    str x28, [sp, 64]
    stp d8, d9, [sp, 72]
    stp d10, d11, [sp, 88]
    stp d12, d13, [sp, 104]
    stp d14, d15, [sp, 120]
    KAI_ASM_INST(0xd503477f)        // SMSTART ZA
    mov x17, #0
    ldr x16, [x0, #0x18]
    ptrue p1.b
    ldr x20, [x0, #0x78]
    cntw x15
    ldr x14, [x0, #0x8]
    add x16, x16, #1
    ldr x13, [x0, #0x10]
    ldr x11, [x0, #0x28]
    lsr x16, x16, #1
    KAI_ASM_INST(0x84c0a681)        // ld1rh { z1.h }, p1/Z, [x20]
    ldr x10, [x0, #0x38]
    KAI_ASM_INST(0x84c1a69c)        // ld1rh { z28.h }, p1/Z, [x20, #2]
    ldr x9, [x0, #0x48]
    ldr x28, [x0, #0x20]
KAI_ASM_LABEL(label_1)              // m_loop
    mov x27, #0
    ldr x26, [x0, #0x30]
KAI_ASM_LABEL(label_2)              // n_loop
    mov x25, x26
    KAI_ASM_INST(0xc00800ff)        // zero { za }
    mov x24, x28
    add x23, x25, x10
    ld1h { z19.s }, p1/Z, [x25]
    ld1h { z0.s }, p1/Z, [x23]
    incw x25, ALL, MUL #2
    incw x23, ALL, MUL #2
    fcvt z19.s, p1/m, z19.h
    fcvt z0.s, p1/m, z0.h
    KAI_ASM_INST(0xc0902660)        // addha za0.s, p1/M, p1/M, z19.s
    KAI_ASM_INST(0xc0902402)        // addha za2.s, p1/M, p1/M, z0.s
    whilelt p0.h, x27, x13
    mov x22, x16
    cmp x22, #2
    blt label_4
KAI_ASM_LABEL(label_3)              // K loop, pair groups
    ld1h { z8.h }, p1/Z, [x24]
    ld1h { z12.h }, p1/Z, [x25]
    ld1h { z16.h }, p1/Z, [x23]
    addvl x24, x24, #1
    addvl x25, x25, #1
    addvl x23, x23, #1
    KAI_ASM_INST(0x81ac2500)        // fmopa za0.s, p1/M, p1/M, z8.h, z12.h
    KAI_ASM_INST(0x81b02502)        // fmopa za2.s, p1/M, p1/M, z8.h, z16.h
    ld1h { z8.h }, p1/Z, [x24]
    ld1h { z12.h }, p1/Z, [x25]
    ld1h { z16.h }, p1/Z, [x23]
    addvl x24, x24, #1
    addvl x25, x25, #1
    addvl x23, x23, #1
    KAI_ASM_INST(0x81ac2501)        // fmopa za1.s, p1/M, p1/M, z8.h, z12.h
    KAI_ASM_INST(0x81b02503)        // fmopa za3.s, p1/M, p1/M, z8.h, z16.h
    sub x22, x22, #2
    cmp x22, #2
    bge label_3
KAI_ASM_LABEL(label_4)              // K tail accumulates in za0/za2
    cbz x22, label_5
    ld1h { z8.h }, p1/Z, [x24]
    ld1h { z12.h }, p1/Z, [x25]
    ld1h { z16.h }, p1/Z, [x23]
    addvl x24, x24, #1
    addvl x25, x25, #1
    addvl x23, x23, #1
    KAI_ASM_INST(0x81ac2500)        // fmopa za0.s, p1/M, p1/M, z8.h, z12.h
    KAI_ASM_INST(0x81b02502)        // fmopa za2.s, p1/M, p1/M, z8.h, z16.h
KAI_ASM_LABEL(label_5)              // K loop done
    ldr x21, [x0, #0x40]
    sub x20, x14, x17
    mov x12, #0
    cmp x20, x15
    csel x20, x20, x15, LS
    madd x21, x9, x17, x21
    add x21, x21, x27, LSL #1
KAI_ASM_LABEL(label_6)              // post-process one output row
    // Pair reductions for the first and second N blocks.
    KAI_ASM_INST(0xc0820408)        // mova z8.s, p1/M, za0h.s[w12, 0]
    KAI_ASM_INST(0xc0820489)        // mova z9.s, p1/M, za1h.s[w12, 0]
    KAI_ASM_INST(0xc082050a)        // mova z10.s, p1/M, za2h.s[w12, 0]
    KAI_ASM_INST(0xc082058b)        // mova z11.s, p1/M, za3h.s[w12, 0]
    add x12, x12, #1
    fadd z8.s, z8.s, z9.s
    fadd z10.s, z10.s, z11.s
    fcvt z8.h, p1/m, z8.s
    fcvt z10.h, p1/m, z10.s
    uzp1 z8.h, z8.h, z10.h
    fmax z8.h, p1/m, z8.h, z1.h
    fmin z8.h, p1/m, z8.h, z28.h
    st1h { z8.h }, p0, [x21]
    add x21, x21, x9
    subs x20, x20, #1
    bne label_6
    incw x27, ALL, MUL #2
    add x26, x26, x10, LSL #1
    cmp x27, x13
    blt label_2
    incw x17
    add x28, x28, x11
    cmp x17, x14
    blt label_1
    KAI_ASM_INST(0xd503467f)        // SMSTOP
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx8vs_qmx_mopa)

    KAI_ASM_END
