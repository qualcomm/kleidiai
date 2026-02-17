//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

.macro scvtf_convert, reg1, reg2, reg3, reg4, pred
scvtf \reg1, \pred/M, \reg1
scvtf \reg2, \pred/M, \reg2
scvtf \reg3, \pred/M, \reg3
scvtf \reg4, \pred/M, \reg4
.endm
.macro clamp_float_4 reg1, reg2, reg3, reg4,min_val, max_val,  pred
FMIN \reg1, \pred/M, \reg1, \max_val
FMIN \reg2, \pred/M, \reg2, \max_val
FMIN \reg3, \pred/M, \reg3, \max_val
FMIN \reg4, \pred/M, \reg4, \max_val
FMAX \reg1, \pred/M, \reg1, \min_val
FMAX \reg2, \pred/M, \reg2, \min_val
FMAX \reg3, \pred/M, \reg3, \min_val
FMAX \reg4, \pred/M, \reg4, \min_val
.endm
.macro ld1b_4p, reg1, reg2, reg3, reg4, pred1, pred2, pred3, pred4,  addr
ld1b \reg1, \pred1/Z, [\addr]
ld1b \reg2, \pred2/Z, [\addr, #1, MUL VL]
ld1b \reg3, \pred3/Z, [\addr, #2, MUL VL]
ld1b \reg4, \pred4/Z, [\addr, #3, MUL VL]
.endm    
.macro ld1w_4p, reg1, reg2, reg3, reg4, pred1, pred2, pred3, pred4,  addr
ld1w \reg1, \pred1/Z, [\addr]
ld1w \reg2, \pred2/Z, [\addr, #1, MUL VL]
ld1w \reg3, \pred3/Z, [\addr, #2, MUL VL]
ld1w \reg4, \pred4/Z, [\addr, #3, MUL VL]
.endm
.macro st1w_4pd reg1, reg2, reg3, reg4, pred1,pred2,pred3,pred4, addr
st1w \reg1, \pred1, [\addr]
st1w \reg2, \pred2, [\addr, #1, MUL VL]
st1w \reg3, \pred3, [\addr, #2, MUL VL]
st1w \reg4, \pred4, [\addr, #3, MUL VL]
.endm
// Extract and interleave 2-bit fields from z0 into d0..d3, then subtract 2
        // Requirements: SVE2
        // Arguments:
        //   d0,d1,d2,d3 : destination/result Z regs (will be overwritten)
        //   s0          : source Z reg (read-only)
        //   t0..t3      : temporary Z regs used for zips (will be clobbered)
        //
        // Example:
        //   EXTRACT_2BIT_QUADS z8, z9, z10, z11, z0, z18, z19, z28, z29
        //
        .macro EXTRACT_2BIT_QUADS d0, d1, d2, d3, s0, t0, t1, t2, t3
            // Duplicate s0 into d0 via ORR, and create right-shifted views
            orr     \d0\().b,  \s0\().b,  \s0\().b
            lsr     \d1\().b,  \s0\().b,  #2
            lsr     \d2\().b,  \s0\().b,  #4
            lsr     \d3\().b,  \s0\().b,  #6

            // Mask to keep only 2 bits in each byte
            and     \d0\().b,  \d0\().b,  #0x03
            and     \d1\().b,  \d1\().b,  #0x03
            and     \d2\().b,  \d2\().b,  #0x03
            and     \d3\().b,  \d3\().b,  #0x03

            // Interleave pairs across .b lanes into temporaries
            zip1    \t0\().b,  \d0\().b,  \d1\().b
            zip2    \t1\().b,  \d0\().b,  \d1\().b
            zip1    \t2\().b,  \d2\().b,  \d3\().b
            zip2    \t3\().b,  \d2\().b,  \d3\().b

            // Interleave across .h lanes to final results
            zip1    \d0\().h,  \t0\().h,  \t2\().h
            zip2    \d1\().h,  \t0\().h,  \t2\().h
            zip1    \d2\().h,  \t1\().h,  \t3\().h
            zip2    \d3\().h,  \t1\().h,  \t3\().h

            // Final bias: subtract 2 from each byte of the outputs
            sub     \d0\().b,  \d0\().b,  #2
            sub     \d1\().b,  \d1\().b,  #2
            sub     \d2\().b,  \d2\().b,  #2
            sub     \d3\().b,  \d3\().b,  #2
        .endm
.macro sdot_4, reg1x, reg2x, reg3x, reg4x, reg1, reg2, reg3, reg4, reg5
sdot \reg1x,  \reg1,  \reg5
sdot \reg2x,  \reg2,  \reg5
sdot \reg3x,  \reg3,  \reg5
sdot \reg4x,  \reg4,  \reg5
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
    KAI_ASM_CODE(matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme1_dot)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme1_dot)
KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme1_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme1_dot)
    stp     x19, x20, [sp, -144]!
    stp     x21, x22, [sp, 16]
    stp     x23, x24, [sp, 32]
    stp     x25, x26, [sp, 48]
    stp     x27, x28, [sp, 64]
    stp     d8, d9,   [sp, 80]
    stp     d10, d11, [sp, 96]
    stp     d12, d13, [sp, 112]
    stp     d14, d15, [sp, 128]
    KAI_ASM_INST(0xd503477f)            // smstart
    mov     x11, #0
    ldr     x15, [x0, 0x38]             // n
    cntw    x19, ALL, MUL #4
    cntw    x9
    ldr     x21, [x0, 0x08]             // lhs_packed

    ptrue   p0.b
    KAI_ASM_INST(0x8555c01e)            // ld1rw   { z30.s }, p0/Z, [x0, 0x54] // min
    KAI_ASM_INST(0x8554c01f)            // ld1rw   { z31.s }, p0/Z, [x0, 0x50] // max
    ldr     x8, [x0, 0x40]              // k
    mul     x9, x9, x8                  // rhs_row_bytes
    ldr     x10, [x0, 0x0]              // dst
    ldr     x12, [x0, 0x30]             // m
KAI_ASM_LABEL(M_Loop)                   // M Loop
    ldr     x26, [x0, 0x10]             // rhs_packed
    mov     x27, x10
    mov     x22, #0
    whilelt p3.s, x22, x15
KAI_ASM_LABEL(N_Loop)                   // N Loop
    mov     x24, x26
    add     x25, x26, x9
    mov     x23, #0
    whilelt p1.b, x23, x8
    dup z4.s, #0
    dup z5.s, #0
    dup z6.s, #0
    dup z7.s, #0
KAI_ASM_LABEL(K_Loop)                   // K Loop
    ld1rqb  { z0.b }, p1/z , [x21, x23]
    whilelt	p4.b, x24, x25
    incb	x24
    whilelt	p5.b, x24, x25
    incb	x24
    whilelt	p6.b, x24, x25
    incb	x24
    whilelt	p7.b, x24, x25
    decb x24 , ALL, MUL #3
    ld1b_4p z16.b,z17.b,z18.b, z19.b, p4,p5,p6,p7,x24
    EXTRACT_2BIT_QUADS z24,z25,z26,z27,z16,z2,z3,z28,z29
    EXTRACT_2BIT_QUADS z20,z21,z22,z23,z17,z2,z3,z28,z29
    // sdot    za.s[w11,0, vgx4], { z24.b - z27.b }, z0.b[0]
    // sdot    za.s[w11,0, vgx4], { z20.b - z23.b }, z0.b[1]
    sdot_4  z4.s,z5.s,z6.s,z7.s, z24.b,z25.b,z26.b,z27.b, z0.b[0]
    sdot_4  z4.s,z5.s,z6.s,z7.s, z20.b,z21.b,z22.b,z23.b, z0.b[1]

    EXTRACT_2BIT_QUADS z8,z9,z10,z11,z18,z2,z3,z28,z29
    EXTRACT_2BIT_QUADS z12,z13,z14,z15,z19,z2,z3,z28,z29
    // sdot    za.s[w11,0, vgx4], { z8.b - z11.b }, z0.b[2]
    // sdot    za.s[w11,0, vgx4], { z12.b - z15.b }, z0.b[3]
    sdot_4  z4.s,z5.s,z6.s,z7.s, z8.b,z9.b,z10.b,z11.b, z0.b[2]
    sdot_4  z4.s,z5.s,z6.s,z7.s, z12.b,z13.b,z14.b,z15.b, z0.b[3]
    addvl   x24, x24, #4
    add     x23, x23, #16
    whilelt p1.b, x23, x8
    b.mi    K_Loop                      // b.first K_Loop
    // Load LHS zero-points & scale factors
    add     x28, x21, x8
    KAI_ASM_INST(0x8540c382)            // ld1rw   { z2.s }, p0/z , [x28]
    KAI_ASM_INST(0x8541c383)            // ld1rw   { z3.s }, p0/z , [x28, #4]
    // Load RHS row sum, scale factors & bias
    add     x28, x26, x9
	whilelt	p4.s, x11, x19
    incw	x11
    whilelt	p5.s, x11, x19
    incw	x11
    whilelt	p6.s, x11, x19
    incw	x11
    whilelt	p7.s, x11, x19
    decw x11 , ALL, MUL #3
	ld1w_4p z8.s,z9.s,z10.s,z11.s,p4,p5,p6,p7,x28
    addvl   x28, x28, #4
	ld1w_4p z24.s,z25.s,z26.s,z27.s,p4,p5,p6,p7,x28
    addvl   x28, x28, #4
	ld1w_4p z12.s,z13.s,z14.s,z15.s,p4,p5,p6,p7,x28
    addvl   x28, x28, #-8
    // Load inner accumulated value
    // LHS scaling factor * RHS scaling factor
    fmul    z24.s, z24.s, z3.s
    fmul    z25.s, z25.s, z3.s
    fmul    z26.s, z26.s, z3.s
    fmul    z27.s, z27.s, z3.s
    // Zero-points x Row-sum + iacc
    mla     z4.s, p0/m, z8.s, z2.s
    mla     z5.s, p0/m, z9.s, z2.s
    mla     z6.s, p0/m, z10.s, z2.s
    mla     z7.s, p0/m, z11.s, z2.s
    scvtf_convert z4.s, z5.s, z6.s, z7.s, p0
    // Apply combined scale factors and add bias
    fmla    z12.s, p0/m, z24.s, z4.s
    fmla    z13.s, p0/m, z25.s, z5.s
    fmla    z14.s, p0/m, z26.s, z6.s
    fmla    z15.s, p0/m, z27.s, z7.s
    // Clamp and store
    clamp_float_4 z12.s,z13.s,z14.s,z15.s, z30.s, z31.s,p0
    incw	x22
    whilelt	p5.s, x22, x15
    incw	x22
    whilelt	p6.s, x22, x15
    incw	x22
    whilelt	p7.s, x22, x15
    decw x22 , ALL, MUL #3
    add x14, x27, x22, lsl #2
    st1w_4pd z12.s,z13.s,z14.s,z15.s, p3,p5,p6,p7, x14
    ldr     x13, [x0, 0x28]             // rhs_stride
    add     x26, x26, x13
    addvl   x22, x22, #1
    whilelt p3.s, x22, x15
    b.lt    N_Loop
    ldr     x13, [x0, 0x18]             // dst_stride_row
    add     x10, x10, x13
    ldr     x13, [x0, 0x20]             // lhs_stride
    add     x21, x21, x13
    subs    x12, x12, #0x1
    bgt     M_Loop
    KAI_ASM_INST(0xd503467f)            // smstop
    ldp     d14, d15, [sp, 128]
    ldp     d12, d13, [sp, 112]
    ldp     d10, d11, [sp, 96]
    ldp     d8, d9,   [sp, 80]
    ldp     x27, x28, [sp, 64]
    ldp     x25, x26, [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp],144
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme1_dot)

    KAI_ASM_END
