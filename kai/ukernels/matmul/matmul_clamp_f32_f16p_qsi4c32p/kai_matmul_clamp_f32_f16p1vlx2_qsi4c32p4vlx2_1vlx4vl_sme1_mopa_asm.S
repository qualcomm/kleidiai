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
    KAI_ASM_CODE(matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme1_mopa)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme1_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme1_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme1_mopa)
    stp     x19, x20, [sp, -128 ]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     d8, d9,   [sp, 48]
    stp     d10, d11, [sp, 64]
    stp     d12, d13, [sp, 80]
    stp     d14, d15, [sp, 96]
    KAI_ASM_INST(0xd503477f)        // smstart
    cntw x14
    ptrue p2.b, all
    ldr x19, [x0, #0x58]            // lut
    ldr z16, [x19]
    uzp1 z16.h, z16.h, z16.h
    //KAI_ASM_INST(0xe11f8260) //ldr zt0, [x19]
    
    // Generate interleave index pattern for tbl instruction
    // Pattern: {0, 128, 64, 128, 1, 128, 65, 128, 2, 128, 66, 128, ...}
    // This interleaves low nibbles (0,1,2,...) with high nibbles (64,65,66,...)
    // with 128 padding between each
    index z13.b, #0, #1          // {0, 1, 2, ...}
    mov z14.d, z13.d
    add z14.b, z14.b, #64        // {64, 65, 66, ...}
    zip1 z15.b, z13.b, z14.b     // {0, 64, 1, 65, 2, 66, ...}
    mov z14.b, #128              // {128, 128, 128, ...}
    zip1 z13.b, z13.b, z14.b     // z13 = {0, 128, 64, 128, 1, 128, 65, 128, 2, 128, 66, 128, ...}

    // Create offset versions of z13 for subsequent tbl instructions
    mov z20.d, z13.d
    mov z21.d, z13.d
    add z21.b, z21.b, #64        // z21 = z13 + 16
    mov z22.d, z13.d
    add z22.b, z22.b, #32        // z22 = z13 + 32
    mov z23.d, z13.d
    add z23.b, z23.b, #96        // z23 = z13 + 48
    
    mov x8, #0
    ldr x16, [x0, #0x10] // rhs_packed
    ldr x17, [x0, #0x18] // rhs_scales
    ldr x6, [x0, #0x40] // N
    KAI_ASM_INST(0x8558c81c) // ld1rw z28.s, p2/z, [x0, #0x60] // min
    KAI_ASM_INST(0x8559c81d) // ld1rw z29.s, p2/z, [x0, #0x64] // max

    //KAI_ASM_INST(0x25664510) // whilelt pn8.h, x8, x6, VLx2
    whilelt p3.h, x8, x6
    //KAI_ASM_INST(0x25a66511) // whilelt pn9.s, x8, x6, VLx4
    whilelt p4.s, x8, x6
    ldr x4, [x0] // dst
    b.eq label_9                    // b.none label_9
KAI_ASM_LABEL(label_1) // N loop
    ldr x9, [x0, #0x38] // M
    KAI_ASM_INST(0x25261501) // whilelt p1.b, x8, x6
    ldr x22, [x0, #0x8] // lhs_packed
    mov x24, x4 // dst
KAI_ASM_LABEL(label_2) // M loop
    KAI_ASM_INST(0xc00800ff) // zero {za}
    KAI_ASM_INST(0x25a917e0)// whilelt p0.s, xzr, x9
    mov x23, x22
    mov x20, #0
    mov x21, #0
    ldr x10, [x0, #0x48] // K
    cmp x10, #0
    b.eq label_8
KAI_ASM_LABEL(label_3) // K loop
    lsl x5, x21, 1
    //KAI_ASM_INST(0xa0052220) // ld1h {z0.h - z1.h}, pn8/z, [x17, x5, lsl #1]
    ld1h z0.h, p3/z, [x17, x5, lsl #1]
    add x13, x17, x5, lsl #1
    inch x8, all
    whilelt p5.h, x8, x6
    dech x8, all
    ld1h z1.h, p5/z, [x13, #1, MUL VL]
    //KAI_ASM_INST(0xc160d018) // zip {z24.h - z25.h}, z0.h, z0.h
    zip1 z24.h, z0.h, z0.h
    zip2 z25.h, z0.h, z0.h
    //KAI_ASM_INST(0xc161d03a) // zip {z26.h - z27.h}, z1.h, z1.h
    zip1 z26.h, z1.h, z1.h
    zip2 z27.h, z1.h, z1.h
    ldr x11, [x0, #0x50] //bl
KAI_ASM_LABEL(label_4) // Block loop
    lsl x5, x20, 1
    ld2h {z4.h, z5.h}, p2/z, [x16, x5, LSL #1]
    inch x20, all
    KAI_ASM_INST(0xa54e42e7) // ld1w {z7.s}, p0/z, [x23, x14, lsl 2]
    //KAI_ASM_INST(0xc08a90ac) // luti4 {z12.h - z15.h}, zt0, z5[0]
    lsr z6.b, z5.b, #4
    and z5.b, z5.b, #0xF
    tbl z12.b, {z5.b, z6.b}, z20.b
    tbl z13.b, {z5.b, z6.b}, z21.b
    tbl z14.b, {z5.b, z6.b}, z22.b
    tbl z15.b, {z5.b, z6.b}, z23.b
    //
    tbl z12.h, {z16.h}, z12.h
    tbl z13.h, {z16.h}, z13.h
    tbl z14.h, {z16.h}, z14.h
    tbl z15.h, {z16.h}, z15.h
    //
    fmul z12.h, z12.h, z24.h
    fmul z13.h, z13.h, z25.h
    fmul z14.h, z14.h, z26.h
    fmul z15.h, z15.h, z27.h
    fmopa za0.s, p2/m, p2/m, z7.h, z12.h
    fmopa za1.s, p2/m, p2/m, z7.h, z13.h
    fmopa za2.s, p2/m, p2/m, z7.h, z14.h
    fmopa za3.s, p2/m, p2/m, z7.h, z15.h
    KAI_ASM_INST(0xa540a2e6)// ld1w {z6.s}, p0/z, [x23]
    add x23, x23, x14, lsl 3
    //KAI_ASM_INST(0xc08a9088) // luti4 {z8.h - z11.h}, zt0, z4[0]
    lsr z5.b, z4.b, #4
    and z4.b, z4.b, #0xF
    tbl z9.b, {z4.b, z5.b}, z20.b
    tbl z10.b, {z4.b, z5.b}, z21.b
    tbl z11.b, {z4.b, z5.b}, z22.b
    tbl z12.b, {z4.b, z5.b}, z23.b
    //
    tbl z9.h, {z16.h}, z9.h
    tbl z10.h, {z16.h}, z10.h
    tbl z11.h, {z16.h}, z11.h
    tbl z12.h, {z16.h}, z12.h
    //
    fmul z9.h,  z9.h, z24.h
    fmul z10.h,  z10.h, z25.h
    fmul z11.h, z11.h, z26.h
    fmul z12.h, z12.h, z27.h
    fmopa za0.s, p2/m, p2/m, z6.h,  z9.h
    fmopa za1.s, p2/m, p2/m, z6.h,  z10.h
    fmopa za2.s, p2/m, p2/m, z6.h, z11.h
    fmopa za3.s, p2/m, p2/m, z6.h, z12.h
    subs x11, x11, #4
    b.gt label_4
    ldr x5, [x0, #0x50] // bl
    subs x10, x10, x5
    inch x21, all
    b.gt label_3
KAI_ASM_LABEL(label_8)
    cmp x9, x14
    csel x15, x9, x14, lo
    lsl x15, x15, #2
    mov w12, #0
KAI_ASM_LABEL(label_5) // store tiles
    //KAI_ASM_INST(0xc0060400) // mova {z0.b - z3.b}, za0h.b[w12, 0:3]
    mova z0.b, p2/m, za0h.b[w12, 0]
    mova z1.b, p2/m, za0h.b[w12, 1]
    mova z2.b, p2/m, za0h.b[w12, 2]
    mova z3.b, p2/m, za0h.b[w12, 3]
    //KAI_ASM_INST(0xc1bdcb80) // fclamp  { z0.s - z3.s}, z28.s, z29.s
    fmin z0.s, p2/m, z0.s, z29.s
    fmax z0.s, p2/m, z0.s, z28.s
    fmin z1.s, p2/m, z1.s, z29.s
    fmax z1.s, p2/m, z1.s, z28.s
    fmin z2.s, p2/m, z2.s, z29.s
    fmax z2.s, p2/m, z2.s, z28.s
    fmin z3.s, p2/m, z3.s, z29.s
    fmax z3.s, p2/m, z3.s, z28.s
    //KAI_ASM_INST(0xa060c700) // st1w {z0.s - z3.s}, pn9, [x24]
    st1w z0.s, p4, [x24]
    incw  x8, all
    whilelt p5.s, x8, x6
    st1w z1.s, p5, [x24, #1, MUL VL]
    incw  x8, all
    whilelt p5.s, x8, x6
    st1w z2.s, p5, [x24, #2, MUL VL]
    incw  x8, all
    whilelt p5.s, x8, x6
    st1w z3.s, p5, [x24, #3, MUL VL]
    decw  x8, all
    decw  x8, all
    decw  x8, all   
    ldr x5, [x0, #0x20] // dst_stride_row
    add x24, x24, x5
    add w12, w12, #4
    cmp w12, w15
    blt label_5
    ldr x5, [x0, #0x28] // offset_stride_l
    add x22, x22, x5
    decw x9, all
    cmp x9, #0
    b.gt label_2
    incb x4, all, mul #4 // dst
    ldr x5, [x0, #0x30] // rhs_packed_stride
    add x16, x16, x5
    add x17, x17, x5
    incb x8, all
    //KAI_ASM_INST(0x25664510) // whilelt pn8.h, x8, x6, VLx2
    whilelt p3.h, x8, x6
    //KAI_ASM_INST(0x25a66511)  // whilelt pn9.s, x8, x6, VLx4
    whilelt p4.s, x8, x6
    b.mi label_1                    // b.first label_1
KAI_ASM_LABEL(label_9)
    KAI_ASM_INST(0xd503467f) // smstop
    ldp     d14, d15, [sp, 96]
    ldp     d12, d13, [sp, 80]
    ldp     d10, d11, [sp, 64]
    ldp     d8, d9,   [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp],128
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f16p1vlx2_qsi4c32p4vlx2_1vlx4vl_sme1_mopa)

    KAI_ASM_END
