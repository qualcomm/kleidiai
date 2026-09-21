//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Identity INT4 LUT:
// LUT[i] = i, for i = 0..15.
//
// Produces:
//   out_lo.b[i] = src.b[i] & 0x0f
//   out_hi.b[i] = src.b[i] >> 4
//
// Pass register names without element suffixes:
//   luti4_identity z4, z5, z12
.macro luti4_identity out_lo, out_hi, src, tmp
    mov     \tmp\().d, \src\().d
    lsr     \tmp\().b, \tmp\().b, #4
    and     \src\().b, \src\().b, #0x0f
    zip1    \out_lo\().b, \src\().b, \tmp\().b
    zip2    \out_hi\().b, \src\().b, \tmp\().b
.endm


// Base SME/SVE replacement for:
// ld1h {z0.h-z3.h}, pn/z, [addr]
.macro ld1h_4 reg0, reg1, reg2, reg3, pred, addr
    ld1h    {\reg0\().h}, \pred/z, [\addr]
    ld1h    {\reg1\().h}, \pred/z, [\addr, #1, mul vl]
    ld1h    {\reg2\().h}, \pred/z, [\addr, #2, mul vl]
    ld1h    {\reg3\().h}, \pred/z, [\addr, #3, mul vl]
.endm


// Base SME/SVE replacement for:
// scvtf {z0.s-z3.s}, {z0.s-z3.s}
.macro scvtf_4 reg0, reg1, reg2, reg3, pred
    scvtf   \reg0\().s, \pred/m, \reg0\().s
    scvtf   \reg1\().s, \pred/m, \reg1\().s
    scvtf   \reg2\().s, \pred/m, \reg2\().s
    scvtf   \reg3\().s, \pred/m, \reg3\().s
.endm


// Base SME/SVE replacement for SME2 FCLAMP on four vectors.
.macro fclamp_4 reg0, reg1, reg2, reg3, minreg, maxreg, pred
    fmax    \reg0\().s, \pred/m, \reg0\().s, \minreg\().s
    fmin    \reg0\().s, \pred/m, \reg0\().s, \maxreg\().s

    fmax    \reg1\().s, \pred/m, \reg1\().s, \minreg\().s
    fmin    \reg1\().s, \pred/m, \reg1\().s, \maxreg\().s

    fmax    \reg2\().s, \pred/m, \reg2\().s, \minreg\().s
    fmin    \reg2\().s, \pred/m, \reg2\().s, \maxreg\().s

    fmax    \reg3\().s, \pred/m, \reg3\().s, \minreg\().s
    fmin    \reg3\().s, \pred/m, \reg3\().s, \maxreg\().s
.endm


// Four predicated stores replacing:
// st1w {z0.s-z3.s}, pn, [addr]
.macro st1w_4 reg0, reg1, reg2, reg3, \
                 pred0, pred1, pred2, pred3, addr
    st1w    {\reg0\().s}, \pred0, [\addr]
    st1w    {\reg1\().s}, \pred1, [\addr, #1, mul vl]
    st1w    {\reg2\().s}, \pred2, [\addr, #2, mul vl]
    st1w    {\reg3\().s}, \pred3, [\addr, #3, mul vl]
.endm
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
    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_qmx_dot)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_qmx_dot)
KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_qmx_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_qmx_dot)
    stp     x19, x20, [sp, -112 ]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     d8, d9,   [sp, 48]
    stp     d10, d11, [sp, 64]
    stp     d12, d13, [sp, 80]
    stp     d14, d15, [sp, 96]
    KAI_ASM_INST(0xd503477f)        // smstart
    ptrue p2.b, all
    fmov  z28.s, #0.0
	
    //KAI_ASM_INST(0x25607810)        // ptrue pn8.h
    //ldr	x9, [x0, #0x38]             // lut
    //KAI_ASM_INST(0xe11f8120)        // ldr zt0, [x9]
	ptrue p7.h
	
    ldr x10, [x0, #0x10]            // rhs_packed
    ldr x5, [x0]                    // dst
    KAI_ASM_INST(0x8550c81d) // ld1rw z29.s, p2/z, [x0, #0x40] // min
    KAI_ASM_INST(0x8551c81e) // ld1rw z30.s, p2/z, [x0, #0x44] // max
    mov x4, #0
    ldr x24, [x0, #0x20]            // n
	
    //KAI_ASM_INST(0x25b86491)        // whilelt	pn9.s, x4, x24, vlx4
	mov x1,x4
    whilelt p3.s, x1, x24
    incw x1
    whilelt p4.s, x1, x24
    incw x1
    whilelt p5.s, x1, x24
    incw x1
    whilelt p6.s, x1, x24

    // The final tail predicate can be empty while p3 still has valid columns.
    // Test p3 explicitly so partially filled 4-vector N tiles are processed.
    ptest p2, p3.b

    ldr x19, [x0, #0x28]            // k
    ldr x20, [x0, #0x30]            // bl
    b.eq  label_5                   // b.none label_5
KAI_ASM_LABEL(label_1)              // N loop
    ldr x21, [x0, #0x8]             // lhs_packed
    mov x23, x10
    dup z24.s, #0
    dup z25.s, #0
    dup z26.s, #0
    dup z27.s, #0
    mov x6, #0
    whilelt p1.s, x6, x19
    b.eq label_4                    // b.none label_4

KAI_ASM_LABEL(label_2)              // K loop

    // The original code cleared ZA here and later copied ZA to z16-z19.
    // Keep the four accumulators directly in Z registers instead.
    dup     z16.s, #0
    dup     z17.s, #0
    dup     z18.s, #0
    dup     z19.s, #0

    mov     x13, x20

KAI_ASM_LABEL(label_3)              // BL loop
    ld1rqb  {z0.b}, p2/z, [x21]
    add     x21, x21, #16

    // Original SME2:
    // ld1h {z12.h-z15.h}, pn8/z, [x23]
    ld1h_4 z12, z13, z14, z15, p7, x23
    addvl   x23, x23, #4

    // Original SME2 LUTI4 instructions.
    luti4_identity z4,  z5,  z12,z20
    luti4_identity z6,  z7,  z13,z20
    luti4_identity z8,  z9,  z14,z20
    luti4_identity z10, z11, z15,z20

    // Original:
    // sdot za.s[w8,0,vgx4], {z4.b-z7.b}, z0.b[0]
    sdot    z16.s, z4.b, z0.b[0]
    sdot    z17.s, z5.b, z0.b[0]
    sdot    z18.s, z6.b, z0.b[0]
    sdot    z19.s, z7.b, z0.b[0]

    // Original:
    // sdot za.s[w8,0,vgx4], {z8.b-z11.b}, z0.b[1]
    sdot    z16.s, z8.b,  z0.b[1]
    sdot    z17.s, z9.b,  z0.b[1]
    sdot    z18.s, z10.b, z0.b[1]
    sdot    z19.s, z11.b, z0.b[1]

    // Load the next four packed RHS vectors.
    ld1h_4 z12, z13, z14, z15, p7, x23
    addvl   x23, x23, #4

    luti4_identity z4,  z5,  z12,z20
    luti4_identity z6,  z7,  z13,z20
    luti4_identity z8,  z9,  z14,z20
    luti4_identity z10, z11, z15,z20

    // Original:
    // sdot za.s[w8,0,vgx4], {z4.b-z7.b}, z0.b[2]
    sdot    z16.s, z4.b, z0.b[2]
    sdot    z17.s, z5.b, z0.b[2]
    sdot    z18.s, z6.b, z0.b[2]
    sdot    z19.s, z7.b, z0.b[2]

    // Original:
    // sdot za.s[w8,0,vgx4], {z8.b-z11.b}, z0.b[3]
    sdot    z16.s, z8.b,  z0.b[3]
    sdot    z17.s, z9.b,  z0.b[3]
    sdot    z18.s, z10.b, z0.b[3]
    sdot    z19.s, z11.b, z0.b[3]

    subs    x13, x13, #16
    b.gt    label_3

    // No MOVA is needed here.
    // z16-z19 already contain the final INT32 accumulators.
	
    KAI_ASM_INST(0x84c0aaa1)        // ld1rh { z1.h }, p2/z, [x21] // lhs sum
    KAI_ASM_INST(0x84c1aaa2)        // ld1rh { z2.h }, p2/z, [x21, #2] // lhs scale
    add x21, x21, #4
    // The QMX payload loads use the ordinary SVE predicate p7, so metadata
    // must use it too. pn8 is an SME2 pair predicate and is not initialized here.
    ld1h    z4.h, p7/z, [x23]                    // rhs offset, first half
    ld1h    z5.h, p7/z, [x23, #1, mul vl]        // rhs offset, second half
    ld1h    z12.h, p7/z, [x23, #2, mul vl]       // rhs scale, first half
    ld1h    z13.h, p7/z, [x23, #3, mul vl]       // rhs scale, second half
    addvl x23, x23, #4
	
    //KAI_ASM_INST(0xc132e210)        // scvtf{z16.s - z19.s}, {z16.s - z19.s}
	scvtf_4 z16, z17, z18, z19, p2
	
    fmlalb z24.s, z1.h, z4.h
    fmlalb z25.s, z1.h, z5.h
    fmlalt z26.s, z1.h, z4.h
    fmlalt z27.s, z1.h, z5.h
    movprfx z4, z28
    fmlalb z4.s, z12.h, z2.h
    movprfx z5, z28
    fmlalb z5.s, z13.h, z2.h
    movprfx z6, z28
    fmlalt z6.s, z12.h, z2.h
    movprfx z7, z28
    fmlalt z7.s, z13.h, z2.h
    fmla z24.s, p2/m, z16.s, z4.s
    fmla z25.s, p2/m, z17.s, z5.s
    fmla z26.s, p2/m, z18.s, z6.s
    fmla z27.s, p2/m, z19.s, z7.s
    add x6, x6, x20
    whilelt p1.s, x6, x19
    b.mi label_2                    // b.first label_2
KAI_ASM_LABEL(label_4)
    //KAI_ASM_INST(0xc1becbb8)        // fclamp  { z24.s - z27.s }, z29.s, z30.s
	fclamp_4 z24, z25, z26, z27, z29, z30, p2
	
    //KAI_ASM_INST(0xa060c4b8)        // st1w { z24.s-z27.s }, pn9, [x5]
	st1w {z24.s}, p3, [x5]
	st1w {z25.s}, p4, [x5, #1, MUL VL]
	st1w {z26.s}, p5, [x5, #2, MUL VL]
	st1w {z27.s}, p6, [x5, #3, MUL VL]
	
    incb  x4, all
    addvl x5, x5, #4
    ldr x22, [x0, #0x18]
    add x10, x10, x22
    //KAI_ASM_INST(0x25b86491)        // whilelt pn9.s, x4, x24, VLx4
	mov x1,x4
    whilelt p3.s, x1, x24
    incw x1
    whilelt p4.s, x1, x24
    incw x1
    whilelt p5.s, x1, x24
    incw x1
    whilelt p6.s, x1, x24

    // Continue while the next panel's first vector has active lanes.
    ptest p2, p3.b
    b.mi  label_1                   // b.first label_1
KAI_ASM_LABEL(label_5)
    KAI_ASM_INST(0xd503467f)        // smstop
    ldp     d14, d15, [sp, 96]
    ldp     d12, d13, [sp, 80]
    ldp     d10, d11, [sp, 64]
    ldp     d8, d9,   [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp],112
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_qmx_dot)

    KAI_ASM_END
