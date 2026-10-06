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

    KAI_ASM_CODE(matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx16vs_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx16vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx16vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx16vs_qmx_mopa)
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

    mov x7, #0
    ldr x8, [x0, #0x18]      // K
    ptrue p0.b
    ldr x20, [x0, #0x78]     // clamp_args_ptr
    cntw x17
    ldr x16, [x0, #0x08]     // M
    add x8, x8, #1
    ldr x15, [x0, #0x10]     // N
    ldr x14, [x0, #0x28]     // lhs_stride_row
    lsr x8, x8, #1           // ceil(K / 2)
    KAI_ASM_INST(0x84c0a291)  // ld1rh { z17.h }, p0/Z, [x20]
    ldr x13, [x0, #0x38]     // rhs_stride_row
    KAI_ASM_INST(0x84c1a294)  // ld1rh { z20.h }, p0/Z, [x20, #2]
    ldr x11, [x0, #0x48]     // dst_stride_row
    ldr x10, [x0, #0x20]     // lhs_ptr

KAI_ASM_LABEL(label_1)  // M loop
    mov x9, #0
    ldr x28, [x0, #0x30]     // rhs_ptr

KAI_ASM_LABEL(label_2)  // N loop
    mov x27, x28              // RHS panel 0
    add x25, x27, x13         // RHS panel 1
    add x24, x27, x13, LSL #1  // RHS panel 2
    add x23, x25, x13, LSL #1  // RHS panel 3
    mov x26, x10
    KAI_ASM_INST(0xc00800ff)  // zero { za }

    // Each panel begins with one FP16 bias vector containing SVL/4 values.
    ld1h { z15.s }, p0/Z, [x27]
    ld1h { z5.s }, p0/Z, [x25]
    ld1h { z23.s }, p0/Z, [x24]
    ld1h { z1.s }, p0/Z, [x23]
    incw x27, ALL, MUL #2
    incw x25, ALL, MUL #2
    incw x24, ALL, MUL #2
    incw x23, ALL, MUL #2
    fcvt z15.s, p0/m, z15.h
    fcvt z5.s, p0/m, z5.h
    fcvt z23.s, p0/m, z23.h
    fcvt z1.s, p0/m, z1.h
    KAI_ASM_INST(0xc09001e0)  // addha za0.s, p0/M, p0/M, z15.s
    KAI_ASM_INST(0xc09000a1)  // addha za1.s, p0/M, p0/M, z5.s
    KAI_ASM_INST(0xc09002e2)  // addha za2.s, p0/M, p0/M, z23.s
    KAI_ASM_INST(0xc0900023)  // addha za3.s, p0/M, p0/M, z1.s

    mov x22, x8
    cbz x22, label_4

KAI_ASM_LABEL(label_3)  // K-pair loop
    ld1h { z10.h }, p0/Z, [x26]
    ld1h { z4.h }, p0/Z, [x27]
    ld1h { z18.h }, p0/Z, [x25]
    ld1h { z19.h }, p0/Z, [x24]
    ld1h { z1.h }, p0/Z, [x23]
    addvl x26, x26, #1
    addvl x27, x27, #1
    addvl x25, x25, #1
    addvl x24, x24, #1
    addvl x23, x23, #1
    KAI_ASM_INST(0x81a40140)  // fmopa za0.s, p0/M, p0/M, z10.h, z4.h
    KAI_ASM_INST(0x81b20141)  // fmopa za1.s, p0/M, p0/M, z10.h, z18.h
    KAI_ASM_INST(0x81b30142)  // fmopa za2.s, p0/M, p0/M, z10.h, z19.h
    KAI_ASM_INST(0x81a10143)  // fmopa za3.s, p0/M, p0/M, z10.h, z1.h
    subs x22, x22, #1
    bne label_3

KAI_ASM_LABEL(label_4)  // Store tile
    ldr x21, [x0, #0x40]     // dst_ptr
    sub x20, x16, x7
    cmp x20, x17
    csel x20, x20, x17, LS
    madd x21, x11, x7, x21
    add x21, x21, x9, LSL #1
    mov x12, #0

    // A packed FP16 vector holds two adjacent SVL/4-wide panels.
    whilelt p4.h, x9, x15
    mov x6, x9
    inch x6
    whilelt p5.h, x6, x15

KAI_ASM_LABEL(label_5)  // Row loop
    // ZA0-ZA3 are the four independent SVL/4-wide RHS panels.
    KAI_ASM_INST(0xc0020004)  // mova z4.b, p0/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020025)  // mova z5.b, p0/M, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020046)  // mova z6.b, p0/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020067)  // mova z7.b, p0/M, za0h.b[w12, 3]
    add x12, x12, #4

    fcvt z4.h, p0/m, z4.s
    fcvt z5.h, p0/m, z5.s
    fcvt z6.h, p0/m, z6.s
    fcvt z7.h, p0/m, z7.s

    fmin z4.h, p0/m, z4.h, z20.h
    fmax z4.h, p0/m, z4.h, z17.h
    fmin z5.h, p0/m, z5.h, z20.h
    fmax z5.h, p0/m, z5.h, z17.h
    fmin z6.h, p0/m, z6.h, z20.h
    fmax z6.h, p0/m, z6.h, z17.h
    fmin z7.h, p0/m, z7.h, z20.h
    fmax z7.h, p0/m, z7.h, z17.h

    uzp1 z4.h, z4.h, z5.h
    uzp1 z6.h, z6.h, z7.h
    st1h { z4.h }, p4, [x21]
    st1h { z6.h }, p5, [x21, #1, MUL VL]
    add x21, x21, x11
    subs x20, x20, #1
    bne label_5

    incw x9, ALL, MUL #4
    add x28, x28, x13, LSL #2
    cmp x9, x15
    blt label_2

    incw x7
    add x10, x10, x14
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16p4vsx2_f16p4vsx2bf16_4vsx16vs_qmx_mopa)

    KAI_ASM_END
