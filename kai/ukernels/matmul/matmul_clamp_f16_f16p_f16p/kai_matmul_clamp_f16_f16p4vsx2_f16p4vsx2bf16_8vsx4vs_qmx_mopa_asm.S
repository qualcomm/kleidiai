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

    KAI_ASM_CODE(matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx4vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx4vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx4vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx4vs_qmx_mopa)
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
    mov x17, #0
    ldr x16, [x0, #0x18]
    cntw x15
    ldr x14, [x0, #0x8]
    add x16, x16, #1
    ldr x13, [x0, #0x10]
    ldr x11, [x0, #0x28]
    lsr x16, x16, #1
    ldr x10, [x0, #0x38]
    ldr x9, [x0, #0x48]
    ldr x20, [x0, #0x78]
    KAI_ASM_INST(0x84c0a684)  // ld1rh { z4.h }, p1/Z, [x20]
    KAI_ASM_INST(0x84c1a685)  // ld1rh { z5.h }, p1/Z, [x20, #2]
    ldr x28, [x0, #0x20]

KAI_ASM_LABEL(label_1)  // M loop
    mov x27, #0
    ldr x26, [x0, #0x30]
KAI_ASM_LABEL(label_2)  // N loop
    mov x25, x26
    mov x24, x28
    add x23, x24, x11
    KAI_ASM_INST(0xc00800ff)  // zero { za }
    whilelt p0.s, x27, x13

    // The packed RHS starts with an FP16 bias vector.  Loading into .s
    // lanes keeps one bias value per FP32 accumulator column.
    ld1h { z1.s }, p1/Z, [x25]
    incw x25, ALL, MUL #2
    fcvt z1.s, p1/M, z1.h
    KAI_ASM_INST(0xc0902420)  // addha za0.s, p1/M, p1/M, z1.s
    KAI_ASM_INST(0xc0902421)  // addha za1.s, p1/M, p1/M, z1.s

    // Each widening FMOPA consumes two original K values, so the loop
    // count is ceil(K / 2).  The packed LHS layout supplies one vector
    // for each row tile per K pair; no ZIP, PN predicate or grouped load
    // is used.
    mov x22, x16
    cbz x22, label_5
KAI_ASM_LABEL(label_3)  // K loop
    ld1h { z12.h }, p1/Z, [x24]
    ld1h { z16.h }, p1/Z, [x23]
    ld1h { z28.h }, p1/Z, [x25]
    addvl x24, x24, #1
    addvl x23, x23, #1
    addvl x25, x25, #1
    KAI_ASM_INST(0x81bc2580)  // fmopa za0.s, p1/M, p1/M, z12.h, z28.h
    KAI_ASM_INST(0x81bc2601)  // fmopa za1.s, p1/M, p1/M, z16.h, z28.h
    subs x22, x22, #1
    bne label_3

KAI_ASM_LABEL(label_4)  // K loop done

KAI_ASM_LABEL(label_5)  // Output
    ldr x21, [x0, #0x40]
    madd x21, x9, x17, x21
    add x21, x21, x27, LSL #1
    madd x20, x15, x9, x21

    // In the ZA .h view each accumulated logical ZA tile is paired with an
    // unused zero tile.  Reduce the adjacent slices once before converting
    // the FP32 results back to FP16.
    sub x1, x14, x17
    cmp x1, x15
    csel x1, x1, x15, LS
    mov x12, #0
    cbz x1, label_7
KAI_ASM_LABEL(label_6)  // Store first row tile
    KAI_ASM_INST(0xc0420402)  // mova z2.h, p1/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc0420423)  // mova z3.h, p1/M, za0h.h[w12, 1]
    add x12, x12, #2
    fadd z0.s, z2.s, z3.s
    fcvt z0.h, p1/M, z0.s
    fmin z0.h, p1/M, z0.h, z5.h
    fmax z0.h, p1/M, z0.h, z4.h
    st1h { z0.s }, p0, [x21]
    add x21, x21, x9
    subs x1, x1, #1
    bne label_6

KAI_ASM_LABEL(label_7)  // Store second row tile if present
    add x2, x17, x15
    cmp x14, x2
    bls label_9
    sub x1, x14, x2
    cmp x1, x15
    csel x1, x1, x15, LS
    mov x12, #0
KAI_ASM_LABEL(label_8)
    KAI_ASM_INST(0xc0420502)  // mova z2.h, p1/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420523)  // mova z3.h, p1/M, za1h.h[w12, 1]
    add x12, x12, #2
    fadd z0.s, z2.s, z3.s
    fcvt z0.h, p1/M, z0.s
    fmin z0.h, p1/M, z0.h, z5.h
    fmax z0.h, p1/M, z0.h, z4.h
    st1h { z0.s }, p0, [x20]
    add x20, x20, x9
    subs x1, x1, #1
    bne label_8

KAI_ASM_LABEL(label_9)
    incw x27
    add x26, x26, x10
    cmp x27, x13
    blt label_2
    incw x17, ALL, MUL #2
    add x28, x28, x11, LSL #1
    cmp x17, x14
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx4vs_qmx_mopa)

    KAI_ASM_END
