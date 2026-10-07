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

    KAI_ASM_CODE(matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_qmx_mopa)
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

    mov x8, #0
    ldr x17, [x0, #0x18]  // K
    ptrue p0.b
    ldr x20, [x0, #0x78]  // clamp_args_ptr
    cntw x16
    ldr x15, [x0, #0x8]   // M
    add x17, x17, #1
    ldr x14, [x0, #0x10]  // N
    ldr x13, [x0, #0x28]  // lhs_stride_row
    lsr x17, x17, #1      // ceil(K / 2)
    KAI_ASM_INST(0x84c0a29c)  // ld1rh { z28.h }, p0/Z, [x20]
    ldr x11, [x0, #0x38]  // rhs_stride_row
    KAI_ASM_INST(0x84c1a29d)  // ld1rh { z29.h }, p0/Z, [x20, #2]
    ldr x10, [x0, #0x48]  // dst_stride_row
    ldr x9, [x0, #0x20]   // lhs_ptr

KAI_ASM_LABEL(label_1)  // M loop
    mov x28, #0
    ldr x27, [x0, #0x30]  // rhs_ptr

KAI_ASM_LABEL(label_2)  // N loop
    mov x26, x27                 // RHS panel 0
    add x24, x26, x11            // RHS panel 1
    mov x25, x9                  // LHS panel 0
    add x23, x25, x13            // LHS panel 1
    KAI_ASM_INST(0xc00800ff)     // zero { za }

    // Each packed RHS panel starts with an FP16 bias vector. Widen each
    // independently and seed the four FP32 accumulator quadrants as
    // za0/za2/za1/za3 = top-left/top-right/bottom-left/bottom-right.
    ld1h { z3.s }, p0/Z, [x26]
    ld1h { z4.s }, p0/Z, [x24]
    incw x26, ALL, MUL #2
    incw x24, ALL, MUL #2
    fcvt z3.s, p0/m, z3.h
    fcvt z4.s, p0/m, z4.h
    KAI_ASM_INST(0xc0900060)  // addha za0.s, p0/M, p0/M, z3.s
    KAI_ASM_INST(0xc0900082)  // addha za2.s, p0/M, p0/M, z4.s
    KAI_ASM_INST(0xc0900061)  // addha za1.s, p0/M, p0/M, z3.s
    KAI_ASM_INST(0xc0900083)  // addha za3.s, p0/M, p0/M, z4.s

    mov x22, x17
    mov x21, #0
    mov x20, #0
    cbz x22, label_4

KAI_ASM_LABEL(label_3)  // K loop, one packed pair at a time
    ld1h { z19.h }, p0/Z, [x25, x21, LSL #1]
    ld1h { z23.h }, p0/Z, [x23, x21, LSL #1]
    ld1h { z17.h }, p0/Z, [x26, x20, LSL #1]
    ld1h { z18.h }, p0/Z, [x24, x20, LSL #1]
    inch x21
    inch x20
    KAI_ASM_INST(0x81b10260)  // fmopa za0.s, p0/M, p0/M, z19.h, z17.h
    KAI_ASM_INST(0x81b20262)  // fmopa za2.s, p0/M, p0/M, z19.h, z18.h
    KAI_ASM_INST(0x81b102e1)  // fmopa za1.s, p0/M, p0/M, z23.h, z17.h
    KAI_ASM_INST(0x81b202e3)  // fmopa za3.s, p0/M, p0/M, z23.h, z18.h
    subs x22, x22, #1
    bne label_3

KAI_ASM_LABEL(label_4)  // Store tile
    ldr x22, [x0, #0x40]  // dst_ptr
    madd x22, x10, x8, x22
    add x22, x22, x28, LSL #1
    madd x21, x16, x10, x22
    whilelt p6.h, x28, x14
    mov x12, #0

    // The first M panel is represented by za0/za2 and is exposed through
    // za0h.  Extract two rows at a time, convert each FP32 vector separately,
    // and use UZP1 only to concatenate the two FP16 N-panel halves.
    cntw x20
KAI_ASM_LABEL(label_5)
    KAI_ASM_INST(0xc0420000)  // mova z0.h, p0/M, za0h.h[w12, 0]
    KAI_ASM_INST(0xc0420021)  // mova z1.h, p0/M, za0h.h[w12, 1]
    KAI_ASM_INST(0xc0420043)  // mova z3.h, p0/M, za0h.h[w12, 2]
    KAI_ASM_INST(0xc0420064)  // mova z4.h, p0/M, za0h.h[w12, 3]
    add x12, x12, #4
    fcvt z0.h, p0/m, z0.s
    fcvt z1.h, p0/m, z1.s
    fcvt z3.h, p0/m, z3.s
    fcvt z4.h, p0/m, z4.s
    uzp1 z2.h, z0.h, z1.h
    uzp1 z5.h, z3.h, z4.h
    fmax z2.h, p0/m, z2.h, z28.h
    fmin z2.h, p0/m, z2.h, z29.h
    st1h { z2.h }, p6, [x22]
    add x22, x22, x10
    fmax z5.h, p0/m, z5.h, z28.h
    fmin z5.h, p0/m, z5.h, z29.h
    st1h { z5.h }, p6, [x22]
    add x22, x22, x10
    subs x20, x20, #2
    bne label_5

    // The second M panel is represented by za1/za3 and is exposed through
    // za1h.  Limit it for a direct tail call and handle the final row alone.
    sub x20, x15, x8
    sub x20, x20, x16
    cmp x20, x16
    csel x20, x20, x16, LS
    cmp x20, #2
    bcc label_7
KAI_ASM_LABEL(label_6)
    KAI_ASM_INST(0xc0420100)  // mova z0.h, p0/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420121)  // mova z1.h, p0/M, za1h.h[w12, 1]
    KAI_ASM_INST(0xc0420143)  // mova z3.h, p0/M, za1h.h[w12, 2]
    KAI_ASM_INST(0xc0420164)  // mova z4.h, p0/M, za1h.h[w12, 3]
    add x12, x12, #4
    fcvt z0.h, p0/m, z0.s
    fcvt z1.h, p0/m, z1.s
    fcvt z3.h, p0/m, z3.s
    fcvt z4.h, p0/m, z4.s
    uzp1 z2.h, z0.h, z1.h
    uzp1 z5.h, z3.h, z4.h
    fmax z2.h, p0/m, z2.h, z28.h
    fmin z2.h, p0/m, z2.h, z29.h
    st1h { z2.h }, p6, [x21]
    add x21, x21, x10
    fmax z5.h, p0/m, z5.h, z28.h
    fmin z5.h, p0/m, z5.h, z29.h
    st1h { z5.h }, p6, [x21]
    add x21, x21, x10
    subs x20, x20, #2
    cmp x20, #2
    bcs label_6

KAI_ASM_LABEL(label_7)
    cbz x20, label_8
    KAI_ASM_INST(0xc0420100)  // mova z0.h, p0/M, za1h.h[w12, 0]
    KAI_ASM_INST(0xc0420121)  // mova z1.h, p0/M, za1h.h[w12, 1]
    fcvt z0.h, p0/m, z0.s
    fcvt z1.h, p0/m, z1.s
    uzp1 z2.h, z0.h, z1.h
    fmax z2.h, p0/m, z2.h, z28.h
    fmin z2.h, p0/m, z2.h, z29.h
    st1h { z2.h }, p6, [x21]

KAI_ASM_LABEL(label_8)
    incw x28, ALL, MUL #2
    add x27, x27, x11, LSL #1
    cmp x28, x14
    blt label_2
    incw x8, ALL, MUL #2
    add x9, x9, x13, LSL #1
    cmp x8, x15
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_8vsx8vs_qmx_mopa)

    KAI_ASM_END
