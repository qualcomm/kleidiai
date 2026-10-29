//
// SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//

// QMX (FEAT_SME) port of kai_matmul_clamp_f32_qsi8d32p1x4sf16_qsi4c32p16vsx4s1s0sf16_i8p_1x16vs_sme2_dot.
// The RHS is packed with kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon: each 32-bit lane holds K+0..3 in the
// low nibbles and K+4..7 in the high nibbles (XOR 0x88, scales pre-multiplied by 1/16), so no zip is needed.
// The SME2-only instructions are replaced as follows:
//   * luti4 (ZT0)              -> lsl #4 / and 0xF0 for the default table, and 0x0f / lsr #4 + tbl for a caller table.
//   * sdot za.s[w8, 0, vgx4]   -> indexed SVE sdot into the Z accumulators z16-z19.
//   * multi-vector ld1h/st1w   -> per-vector loads/stores, p3-p6 column predicates.
//   * zip/mova/scvtf/fclamp x4 -> single-vector zip1/zip2 (once per block), scvtf, fmax/fmin.

// Loads 8 K values of RHS into z12-z15.
.macro load_rhs_k8
    ld1w { z12.s }, p2/z, [x0]
    ld1w { z13.s }, p2/z, [x0, #1, mul vl]
    ld1w { z14.s }, p2/z, [x0, #2, mul vl]
    ld1w { z15.s }, p2/z, [x0, #3, mul vl]
    addvl x0, x0, #4
.endm

// Default table: K+0..3 -> z4, z6, z8, z10 and K+4..7 -> z5, z7, z9, z11. Requires z29.b = 0xf0.
.macro decode_k8
    lsl z4.b, z12.b, #4
    and z5.d, z12.d, z29.d
    lsl z6.b, z13.b, #4
    and z7.d, z13.d, z29.d
    lsl z8.b, z14.b, #4
    and z9.d, z14.d, z29.d
    lsl z10.b, z15.b, #4
    and z11.d, z15.d, z29.d
.endm

// Caller table held in z31. Same outputs as decode_k8. Requires z30.b = 0x0f.
.macro decode_k8_lut
    and z4.d, z12.d, z30.d
    lsr z5.b, z12.b, #4
    and z6.d, z13.d, z30.d
    lsr z7.b, z13.b, #4
    and z8.d, z14.d, z30.d
    lsr z9.b, z14.b, #4
    and z10.d, z15.d, z30.d
    lsr z11.b, z15.b, #4
    tbl z4.b, { z31.b }, z4.b
    tbl z5.b, { z31.b }, z5.b
    tbl z6.b, { z31.b }, z6.b
    tbl z7.b, { z31.b }, z7.b
    tbl z8.b, { z31.b }, z8.b
    tbl z9.b, { z31.b }, z9.b
    tbl z10.b, { z31.b }, z10.b
    tbl z11.b, { z31.b }, z11.b
.endm

.macro sdot_k8 lo, hi
    sdot z16.s, z4.b, z0.b[\lo]
    sdot z17.s, z6.b, z0.b[\lo]
    sdot z18.s, z8.b, z0.b[\lo]
    sdot z19.s, z10.b, z0.b[\lo]
    sdot z16.s, z5.b, z0.b[\hi]
    sdot z17.s, z7.b, z0.b[\hi]
    sdot z18.s, z9.b, z0.b[\hi]
    sdot z19.s, z11.b, z0.b[\hi]
.endm

// Replaces: whilelt pn9.s, start, end, VLx4 (p3 holds the first vector)
.macro whilelt_x4 start, end
    mov x11, \start
    whilelt p3.s, x11, \end
    incw x11
    whilelt p4.s, x11, \end
    incw x11
    whilelt p5.s, x11, \end
    incw x11
    whilelt p6.s, x11, \end
    // The tail predicates can be empty while p3 still has valid columns.
    ptest p2, p3.b
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

    KAI_ASM_CODE(matmul_clamp_f32_qsi8d32p1x4sf16_qsi4c32p16vsx4s4s0sf16_i8p_1x16vs_qmx_dot)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4sf16_qsi4c32p16vsx4s4s0sf16_i8p_1x16vs_qmx_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_qsi8d32p1x4sf16_qsi4c32p16vsx4s4s0sf16_i8p_1x16vs_qmx_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_qsi8d32p1x4sf16_qsi4c32p16vsx4s4s0sf16_i8p_1x16vs_qmx_dot)
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    mov x10, x0
    ldr x9, [x10, #72]
    KAI_ASM_INST(0xd503477f)  // smstart
    ptrue p2.b, all
    fmov z28.s, #0.0

    // x9 == 0: default {-8..7} table. Otherwise x9 points to a 16-byte table indexed by the packed nibble.
    // z23 restores the 1/16 RHS scale pre-scaling, which only the default table compensates for.
    dup z29.b, #0xf0
    fmov z23.h, #1.0
    cbz x9, 1f
    ld1rqb { z31.b }, p2/z, [x9]
    dup z30.b, #0x0f
    fmov z23.h, #16.0
1:
    KAI_ASM_INST(0x8554c954)  // ld1rw { z20.s }, p2/z, [x10, #80]
    KAI_ASM_INST(0x8555c955)  // ld1rw { z21.s }, p2/z, [x10, #84]
    ldr x0, [x10, #8]
    ldr x1, [x10, #16]
    ldr x5, [x10]
    ldr x12, [x10, #56]
    ldr x14, [x10, #64]
    ldr x15, [x10, #40]
    mov x4, #0
    ldr x17, [x10, #48]
    whilelt_x4 x4, x17
    b.none label_n_end
    // N_LOOP START
KAI_ASM_LABEL(label_n_start)
    ldr x2, [x10, #24]
    ldr x3, [x10, #32]
    dup z24.s, #0
    dup z25.s, #0
    dup z26.s, #0
    dup z27.s, #0
    mov x6, #0
    whilelt p1.s, x6, x12
    b.none label_k_end
    // K_LOOP START
KAI_ASM_LABEL(label_k_start)
    dup z16.s, #0
    dup z17.s, #0
    dup z18.s, #0
    dup z19.s, #0
    mov x13, x14
    cbnz x9, label_bl_lut_start
    // BL_LOOP START
KAI_ASM_LABEL(label_bl_start)
    ld1rqb { z0.b }, p2/z, [x2]
    add x2, x2, #16
    load_rhs_k8
    decode_k8
    sdot_k8 0, 1
    load_rhs_k8
    decode_k8
    sdot_k8 2, 3
    subs x13, x13, #16
    b.gt label_bl_start
    b label_bl_end
    // BL_LOOP END
    // BL_LOOP (caller table) START
KAI_ASM_LABEL(label_bl_lut_start)
    ld1rqb { z0.b }, p2/z, [x2]
    add x2, x2, #16
    load_rhs_k8
    decode_k8_lut
    sdot_k8 0, 1
    load_rhs_k8
    decode_k8_lut
    sdot_k8 2, 3
    subs x13, x13, #16
    b.gt label_bl_lut_start
    // BL_LOOP (caller table) END
KAI_ASM_LABEL(label_bl_end)
    scvtf z16.s, p2/m, z16.s
    scvtf z17.s, p2/m, z17.s
    scvtf z18.s, p2/m, z18.s
    scvtf z19.s, p2/m, z19.s
    KAI_ASM_INST(0x84c0a861)  // ld1rh { z1.h }, p2/z, [x3]
    add x3, x3, #2
    fmul z1.h, z1.h, z23.h
    ld1h { z4.h }, p2/z, [x1]
    ld1h { z5.h }, p2/z, [x1, #1, mul vl]
    addvl x1, x1, #2
    zip1 z2.h, z4.h, z5.h
    zip2 z3.h, z4.h, z5.h
    movprfx z4, z28
    fmlalb z4.s, z1.h, z2.h
    movprfx z5, z28
    fmlalb z5.s, z1.h, z3.h
    movprfx z6, z28
    fmlalt z6.s, z1.h, z2.h
    movprfx z7, z28
    fmlalt z7.s, z1.h, z3.h
    fmla z24.s, p2/m, z16.s, z4.s
    fmla z25.s, p2/m, z17.s, z5.s
    fmla z26.s, p2/m, z18.s, z6.s
    fmla z27.s, p2/m, z19.s, z7.s
    add x6, x6, x14
    whilelt p1.s, x6, x12
    b.mi label_k_start
    // K_LOOP END
KAI_ASM_LABEL(label_k_end)
    fmax z24.s, p2/m, z24.s, z20.s
    fmax z25.s, p2/m, z25.s, z20.s
    fmax z26.s, p2/m, z26.s, z20.s
    fmax z27.s, p2/m, z27.s, z20.s
    fmin z24.s, p2/m, z24.s, z21.s
    fmin z25.s, p2/m, z25.s, z21.s
    fmin z26.s, p2/m, z26.s, z21.s
    fmin z27.s, p2/m, z27.s, z21.s
    st1w { z24.s }, p3, [x5]
    st1w { z25.s }, p4, [x5, #1, mul vl]
    st1w { z26.s }, p5, [x5, #2, mul vl]
    st1w { z27.s }, p6, [x5, #3, mul vl]
    incb x4, all
    addvl x5, x5, #4
    mov x7, x0
    mov x0, x1
    add x1, x7, x15
    whilelt_x4 x4, x17
    b.mi label_n_start
    // N_LOOP END
KAI_ASM_LABEL(label_n_end)
    KAI_ASM_INST(0xd503467f)  // smstop
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8, d9, [sp], #64
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_qsi8d32p1x4sf16_qsi4c32p16vsx4s4s0sf16_i8p_1x16vs_qmx_dot)

    KAI_ASM_END
