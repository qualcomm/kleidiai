//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
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

    KAI_ASM_CODE(matmul_clamp_f16_f16p1vlx2_qai4c32p4vlx4_1vlx4vl_sme2_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_f16p1vlx2_qai4c32p4vlx2_1vlx4vl_sme2_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_f16p1vlx2_qai4c32p4vlx2_1vlx4vl_sme2_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_f16p1vlx2_qai4c32p4vlx2_1vlx4vl_sme2_mopa)
    stp     x19, x20, [sp, -128 ]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     x25, x26, [sp, 48]
    stp     d8, d9,   [sp, 64]
    stp     d10, d11, [sp, 80]
    stp     d12, d13, [sp, 96]
    stp     d14, d15, [sp, 112]
    KAI_ASM_INST(0xd503477f)        // smstart
    cntw x14
    ptrue p0.b, all

    ldr x19, [x0, #0x10]            // rhs_packed
    KAI_ASM_INST(0x8558c009)       // ld1rw z9.s, p0/z, [x0, #0x60]  // min
    KAI_ASM_INST(0x8559c00a)       // ld1rw z10.s, p0/z, [x0, #0x64] //max
    fmov z11.h, #0.0                // Used for ?
    ldr x4, [x0, #0x50]             // bl
    ldr x21, [x0, #0x18]            // dst_stride_row
    ldr x20, [x0]                   // dst
    mov x8, #0
    ldr x13, [x0, #0x40]           // n
    ldr x23, [x0, #0x48]            // k
    // Outer N loop
    whilelt p2.h, x8, x13
    inch x8, all
    whilelt p5.h, x8, x13
    dech x8, all
    b.eq label_9                   // b.none label_9
KAI_ASM_LABEL(label_1)              // N Loop
    ldr x9, [x0, #0x38]             // m
    ldr x22, [x0, #0x8]             // lhs_packed
    mov x24, x20                    // dst

KAI_ASM_LABEL(label_2)              // M Loop
    mov x26, x19                    // rhs_ptr reset
    mov x3, x22                     // lhs_ptr
    cmp x9, x14
    csel x15, x9, x14, lo
    lsl x15, x15, #2
    ldr	x10, [x0, #0x48]           // k
    cmp x10, #0
    b.eq label_8
KAI_ASM_LABEL(label_3)              // K Loop
    KAI_ASM_INST(0xc00800ff)        // zero {za}
    mov x11, x4                     // bl counter
    // Load zp and scales for this block (lookahead) using SME1
    addvl x5, x26, #16
    ld1w { z20.s }, p0/z, [x5]
    ld1w { z21.s }, p0/z, [x5, #1, mul vl]
    ld1w { z22.s }, p0/z, [x5, #2, mul vl]
    ld1w { z23.s }, p0/z, [x5, #3, mul vl]
    ld1w { z24.s }, p0/z, [x5, #4, mul vl]
    ld1w { z25.s }, p0/z, [x5, #5, mul vl]
    ld1w { z26.s }, p0/z, [x5, #6, mul vl]
    ld1w { z27.s }, p0/z, [x5, #7, mul vl]

    // Convert zp/scales to f16 (4 vectors each) using SME1
    fcvt z20.h, p0/m, z20.s
    fcvt z16.h, p0/m, z21.s
    uzp1 z16.h, z20.h, z16.h
    fcvt z22.h, p0/m, z22.s
    fcvt z17.h, p0/m, z23.s
    uzp1 z17.h, z22.h, z17.h

    fcvt z24.h, p0/m, z24.s
    fcvt z22.h, p0/m, z25.s
    uzp1 z22.h, z24.h, z22.h
    fcvt z26.h, p0/m, z26.s
    fcvt z23.h, p0/m, z27.s
    uzp1 z23.h, z26.h, z23.h

KAI_ASM_LABEL(label_4)              // BL Loop
    ld2b {z4.b, z5.b}, p0/z, [x26]

    // K Step 0 - Load LHS using SME1
    ld1h { z0.h }, p0/z, [x3]
    ld1h { z1.h }, p0/z, [x3, #1, mul vl]
    
    // Dequantize z4 (4 vectors RHS) - replaces luti4 {z4.h - z7.h}, zt0, z4[0]
    mov z18.d, z4.d                 // Copy z4 to z18 for lower nibbles
    lsr z19.b, z4.b, #4             // Extract upper nibbles (bits 7:4) from z4
    and z18.b, z18.b, #0x0F         // Extract lower nibbles (bits 3:0)
    zip1 z20.b, z18.b, z19.b        // Interleave lower and upper nibbles (lower half)
    zip2 z21.b, z18.b, z19.b        // Interleave lower and upper nibbles (upper half)
    uunpklo z4.h, z20.b             // Unpack lower part of z20
    uunpkhi z5.h, z20.b             // Unpack upper part of z20
    uunpklo z6.h, z21.b             // Unpack lower part of z21
    uunpkhi z7.h, z21.b             // Unpack upper part of z21
    sub z4.h, z4.h, #8              // Convert from [0,15] to [-8,7]
    sub z5.h, z5.h, #8
    sub z6.h, z6.h, #8
    sub z7.h, z7.h, #8
    scvtf z4.h, p2/m, z4.h          // Convert to FP16
    scvtf z5.h, p2/m, z5.h
    scvtf z6.h, p2/m, z6.h
    scvtf z7.h, p2/m, z7.h
    
    // Dequantize z5 from ld2b (4 vectors RHS) - replaces luti4 {z12.h - z15.h}, zt0, z5[0]
    ld2b {z4.b, z5.b}, p0/z, [x26]  // Re-load z5 from original position. TODO: Avoid this.
    addvl x26, x26, #2
    mov z18.d, z28.d                // Copy z5 (now in z28) to z18 for lower nibbles
    lsr z19.b, z28.b, #4            // Extract upper nibbles (bits 7:4) from z5
    and z18.b, z18.b, #0x0F         // Extract lower nibbles (bits 3:0)
    zip1 z20.b, z18.b, z19.b        // Interleave lower and upper nibbles (lower half)
    zip2 z21.b, z18.b, z19.b        // Interleave lower and upper nibbles (upper half)
    uunpklo z12.h, z20.b            // Unpack lower part of z20
    uunpkhi z13.h, z20.b            // Unpack upper part of z20
    uunpklo z14.h, z21.b            // Unpack lower part of z21
    uunpkhi z15.h, z21.b            // Unpack upper part of z21
    sub z12.h, z12.h, #8            // Convert from [0,15] to [-8,7]
    sub z13.h, z13.h, #8
    sub z14.h, z14.h, #8
    sub z15.h, z15.h, #8
    scvtf z12.h, p2/m, z12.h        // Convert to FP16
    scvtf z13.h, p2/m, z13.h
    scvtf z14.h, p2/m, z14.h
    scvtf z15.h, p2/m, z15.h
    
    addvl x3, x3, #2

    // Apply Scale/Offset
    fmul z4.h, z4.h, z22.h
    fadd z4.h, z4.h, z16.h
    fmul z5.h, z5.h, z23.h
    fadd z5.h, z5.h, z17.h
    fmul z6.h, z6.h, z22.h
    fadd z6.h, z6.h, z16.h
    fmul z7.h, z7.h, z23.h
    fadd z7.h, z7.h, z17.h

    // FMOPA (Step 0)
    fmopa za0.s, p0/m, p0/m, z0.h, z4.h
    fmopa za1.s, p0/m, p0/m, z0.h, z5.h
    fmopa za2.s, p0/m, p0/m, z0.h, z6.h
    fmopa za3.s, p0/m, p0/m, z0.h, z7.h

    // K Step 1
    fmul z12.h, z12.h, z22.h
    fadd z12.h, z12.h, z16.h
    fmul z13.h, z13.h, z23.h
    fadd z13.h, z13.h, z17.h
    fmul z14.h, z14.h, z22.h
    fadd z14.h, z14.h, z16.h
    fmul z15.h, z15.h, z23.h
    fadd z15.h, z15.h, z17.h

    // FMOPA (Step 1) with LHS z1
    fmopa za0.s, p0/m, p0/m, z1.h, z12.h
    fmopa za1.s, p0/m, p0/m, z1.h, z13.h
    fmopa za2.s, p0/m, p0/m, z1.h, z14.h
    fmopa za3.s, p0/m, p0/m, z1.h, z15.h

    subs x11, x11, #4
    b.gt label_4

    // Advance past scales/offsets
    addvl x26, x26, #16

    subs x10, x10, x4
    b.gt label_3

KAI_ASM_LABEL(label_8)
    ldr	x5, [x0, #0x30]             // bias offset
    add x5, x5, x19
    // Load bias using SME1
    ld1w { z12.s }, p0/z, [x5]
    ld1w { z13.s }, p0/z, [x5, #1, mul vl]
    ld1w { z14.s }, p0/z, [x5, #2, mul vl]
    ld1w { z15.s }, p0/z, [x5, #3, mul vl]
    mov x12, 0

KAI_ASM_LABEL(label_10)             // Output Loop
    mova z24.s, p0/m, za0h.s[w12, 0]
    mova z25.s, p0/m, za1h.s[w12, 0]
    mova z26.s, p0/m, za2h.s[w12, 0]
    mova z27.s, p0/m, za3h.s[w12, 0]

    // Add Bias
    fadd z24.s, p0/m, z24.s, z12.s
    fadd z25.s, p0/m, z25.s, z13.s
    fadd z26.s, p0/m, z26.s, z14.s
    fadd z27.s, p0/m, z27.s, z15.s

    // Clamp using SME1 single-vector min/max
    fmax z24.s, p0/m, z24.s, z9.s
    fmin z24.s, p0/m, z24.s, z10.s
    fmax z25.s, p0/m, z25.s, z9.s
    fmin z25.s, p0/m, z25.s, z10.s
    fmax z26.s, p0/m, z26.s, z9.s
    fmin z26.s, p0/m, z26.s, z10.s
    fmax z27.s, p0/m, z27.s, z9.s
    fmin z27.s, p0/m, z27.s, z10.s

    // Convert to f16 using SME1
    fcvt z24.h, p0/m, z24.s
    fcvt z28.h, p0/m, z25.s
    uzp1 z28.h, z24.h, z28.h
    fcvt z26.h, p0/m, z26.s
    fcvt z29.h, p0/m, z27.s
    uzp1 z29.h, z26.h, z29.h

    // Store
    st1h z28.h, p2, [x24]
    st1h z29.h, p5, [x24, #1, MUL VL]
    add x24, x24, x21
    add x12, x12, #4
    cmp x12, x15
    blt label_10
    ldr	x5, [x0, #0x20]
    add x22, x22, x5
    mov x24, x25
    decw x9, all
    cmp x9, #0
    b.gt label_2
    incb x20, all, mul #2
    ldr	x5, [x0, #0x28]
    add x19, x19, x5
    incb x8, all
    whilelt p2.h, x8, x13
    inch x8, all
    whilelt p5.h, x8, x13
    dech x8, all
    b.mi label_1                    // b.first label_1
KAI_ASM_LABEL(label_9)
    KAI_ASM_INST(0xd503467f)        // smstop
    ldp     d14, d15, [sp, 112]
    ldp     d12, d13, [sp, 96]
    ldp     d10, d11, [sp, 80]
    ldp     d8, d9,   [sp, 64]
    ldp     x25, x26, [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp],128
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_f16p1vlx2_qai4c32p4vlx2_1vlx4vl_sme2_mopa)
    KAI_ASM_END
