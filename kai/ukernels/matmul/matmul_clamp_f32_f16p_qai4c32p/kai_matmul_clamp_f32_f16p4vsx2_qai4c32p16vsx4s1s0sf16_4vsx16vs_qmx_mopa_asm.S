//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
    KAI_ASM_CODE(matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_qmx_mopa)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_qmx_mopa)
    stp     x19, x20, [sp, -128 ]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     x25, x26, [sp, 48]
    stp     d8, d9,   [sp, 64]
    stp     d10, d11, [sp, 80]
    stp     d12, d13, [sp, 96]
    stp     d14, d15, [sp, 112]
    KAI_ASM_INST(0xd503477f)         // smstart
    cntw x14
    ptrue p2.b, all
    mov x8, #0
    ldr x16, [x0, #0x10]             // rhs_packed
    ldr x6, [x0, #0x38]              // n
    ldr x21, [x0, #0x18]             // dst_stride_row
    ldr x23, [x0, #0x40]             // k
    KAI_ASM_INST(0x8556c81c)         // ld1rw z28.s, p2/z, [x0, #0x58] // min
    KAI_ASM_INST(0x8557c81d)         // ld1rw z29.s, p2/z, [x0, #0x5c] // max

    ptrue p7.h
	mov x1,x8
    whilelt p3.s, x1, x6
    incw x1
    whilelt p4.s, x1, x6
    incw x1
    whilelt p5.s, x1, x6
    incw x1
    whilelt p6.s, x1, x6
	
    ldr x4, [x0]                     // dst
    cmp x8, x6                       // is there any n work left? (whilelt p6 flags are stale/empty when n < 3*VL)
    b.ge label_9                     // b.none label_9
KAI_ASM_LABEL(label_1)               // N loop
    ldr x9, [x0, #0x30]              // m
    KAI_ASM_INST(0x25261501)         // whilelt p1.b, x8, x6
    ldr x22, [x0, #0x8]              // lhs_packed
    mov x24, x4                      // dst tile base
KAI_ASM_LABEL(label_2)               // M loop
    KAI_ASM_INST(0xc00800ff)         // zero {za}
    KAI_ASM_INST(0x25a917e0)         // whilelt p0.s, xzr, x9
    mov x26, x16                     // rhs block pointer
    mov x3, x22                      // lhs block pointer
    cmp x9, x14
    csel x15, x9, x14, lo
    lsl x15, x15, #2
    ldr x10, [x0, #0x40]             // k remaining
    cmp x10, #0
    b.eq label_8
KAI_ASM_LABEL(label_3)               // K loop
    mov x20, #0
    ldr x11, [x0, #0x48]             // bl
    addvl x17, x26, #16              // rhs metadata
    ld1h z0.h, p7/Z, [x17]
    ld1h z1.h, p7/Z, [x17, #1, MUL VL]       // rhs offset, packed in q4k nrx4 order
	addvl x1, x17, #2
    ld1h z2.h, p7/Z, [x1]
    ld1h z3.h, p7/Z, [x1, #1, MUL VL]        // rhs scale, packed in q4k nrx4 order
    uzp1    z20.h, z0.h, z1.h        // restore linear quarter-group order for za0..za3
    uzp2    z21.h, z0.h, z1.h
    uzp1    z22.h, z2.h, z3.h
    uzp2    z23.h, z2.h, z3.h
    zip1    z24.h, z20.h, z20.h      // duplicate offsets for low/high nibble expansion
    zip2    z25.h, z20.h, z20.h
    zip1    z26.h, z21.h, z21.h
    zip2    z27.h, z21.h, z21.h
    zip1    z0.h, z22.h, z22.h       // duplicate scales for low/high nibble expansion
    zip2    z1.h, z22.h, z22.h
    zip1    z2.h, z23.h, z23.h
    zip2    z3.h, z23.h, z23.h
KAI_ASM_LABEL(label_4)               // Block loop
    lsl x5, x20, #2
    KAI_ASM_INST(0xa425c744)         // ld2b {z4.b - z5.b}, p1/z, [x26, x5]
    inch x20, all
     .inst 0xa540a066                        // ld1w {z6.s}, p0/z, [x3]
    mov     z30.d, z4.d
    lsr     z30.b, z30.b, #4
    and     z4.b, z4.b, #0x0f
    zip1    z8.b, z4.b, z30.b
    zip2    z10.b, z4.b, z30.b
    uunpkhi z9.h, z8.b
    uunpklo z8.h, z8.b
    uunpkhi z11.h, z10.b
    uunpklo z10.h, z10.b
    ucvtf   z8.h, p2/m, z8.h
    ucvtf   z9.h, p2/m, z9.h
    ucvtf   z10.h, p2/m, z10.h
    ucvtf   z11.h, p2/m, z11.h
    mov     z16.d, z24.d
    mov     z17.d, z25.d
    mov     z18.d, z26.d
    mov     z19.d, z27.d
    fmla    z16.h, p2/m, z8.h, z0.h
    fmla    z17.h, p2/m, z9.h, z1.h
    fmla    z18.h, p2/m, z10.h, z2.h
    fmla    z19.h, p2/m, z11.h, z3.h
    KAI_ASM_INST(0x81b048c0)         // fmopa za0.s, p2/m, p2/m, z6.h, z16.h
    KAI_ASM_INST(0x81b148c1)         // fmopa za1.s, p2/m, p2/m, z6.h, z17.h
    KAI_ASM_INST(0x81b248c2)         // fmopa za2.s, p2/m, p2/m, z6.h, z18.h
    KAI_ASM_INST(0x81b348c3)         // fmopa za3.s, p2/m, p2/m, z6.h, z19.h
    KAI_ASM_INST(0xa54e4067)         // ld1w {z7.s}, p0/z, [x3, x14, lsl #2]
    add x3, x3, x14, lsl #3
    mov     z30.d, z5.d
    lsr     z30.b, z30.b, #4
    and     z5.b, z5.b, #0x0f
    zip1    z12.b, z5.b, z30.b
    zip2    z14.b, z5.b, z30.b
    uunpkhi z13.h, z12.b
    uunpklo z12.h, z12.b
    uunpkhi z15.h, z14.b
    uunpklo z14.h, z14.b
    ucvtf   z12.h, p2/m, z12.h
    ucvtf   z13.h, p2/m, z13.h
    ucvtf   z14.h, p2/m, z14.h
    ucvtf   z15.h, p2/m, z15.h
    mov     z16.d, z24.d
    mov     z17.d, z25.d
    mov     z18.d, z26.d
    mov     z19.d, z27.d
    fmla    z16.h, p2/m, z12.h, z0.h
    fmla    z17.h, p2/m, z13.h, z1.h
    fmla    z18.h, p2/m, z14.h, z2.h
    fmla    z19.h, p2/m, z15.h, z3.h
    KAI_ASM_INST(0x81b048e0)         // fmopa za0.s, p2/m, p2/m, z7.h, z16.h
    KAI_ASM_INST(0x81b148e1)         // fmopa za1.s, p2/m, p2/m, z7.h, z17.h
    KAI_ASM_INST(0x81b248e2)         // fmopa za2.s, p2/m, p2/m, z7.h, z18.h
    KAI_ASM_INST(0x81b348e3)         // fmopa za3.s, p2/m, p2/m, z7.h, z19.h
    subs x11, x11, #4
    b.gt label_4
    addvl x26, x17, #4               // next rhs block
    ldr x11, [x0, #0x48]             // bl
    subs x10, x10, x11
    b.gt label_3
KAI_ASM_LABEL(label_8)               // store tiles
    mov w12, #0
KAI_ASM_LABEL(label_5)                                            // ZA row loop
    mova z0.b, p2/M, za0h.b[w12, 0]
    mova z1.b, p2/M, za0h.b[w12, 1]
    mova z2.b, p2/M, za0h.b[w12, 2]
    mova z3.b, p2/M, za0h.b[w12, 3]
    FMIN z0.s, p2/M, z0.s, z29.s
    FMIN z1.s, p2/M, z1.s, z29.s
    FMIN z2.s, p2/M, z2.s, z29.s
    FMIN z3.s, p2/M, z3.s, z29.s
    FMAX z0.s, p2/M, z0.s, z28.s
    FMAX z1.s, p2/M, z1.s, z28.s
    FMAX z2.s, p2/M, z2.s, z28.s
    FMAX z3.s, p2/M, z3.s, z28.s
	st1w {z0.s}, p3, [x24]
	st1w {z1.s}, p4, [x24, #1, MUL VL]
	st1w {z2.s}, p5, [x24, #2, MUL VL]
	st1w {z3.s}, p6, [x24, #3, MUL VL]
    add x24, x24, x21
    add w12, w12, #4
    cmp w12, w15
    blt label_5
KAI_ASM_LABEL(label_10)              // next M tile
    ldr x5, [x0, #0x20]              // lhs_packed_stride
    add x22, x22, x5
    decw x9, all
    cmp x9, #0
    b.gt label_2
    incb x4, all, mul #4             // dst
    ldr x5, [x0, #0x28]              // rhs_packed_stride
    add x16, x16, x5
    incb x8, all
	mov x1,x8
    whilelt p3.s, x1, x6
    incw x1
    whilelt p4.s, x1, x6
    incw x1
    whilelt p5.s, x1, x6
    incw x1
    whilelt p6.s, x1, x6

    cmp x8, x6                       // is there any n work left? (whilelt p6 flags are stale/empty when n < 3*VL)
    b.lt label_1                     // b.first label_1
KAI_ASM_LABEL(label_9)
    KAI_ASM_INST(0xd503467f)         // smstop
    ldp     d14, d15, [sp, 112]
    ldp     d12, d13, [sp, 96]
    ldp     d10, d11, [sp, 80]
    ldp     d8, d9,   [sp, 64]
    ldp     x25, x26, [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_qmx_mopa)
    
    KAI_ASM_END
