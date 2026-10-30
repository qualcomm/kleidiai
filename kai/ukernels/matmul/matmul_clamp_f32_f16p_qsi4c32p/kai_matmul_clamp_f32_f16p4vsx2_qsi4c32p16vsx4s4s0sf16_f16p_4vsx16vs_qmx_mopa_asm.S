//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//

// QMX (FEAT_SME) port of kai_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s1s0sf16_f16p_4vsx16vs_sme2_mopa.
// The RHS is packed with kai_rhs_pack_nxk_qsi4c32ps4s0sf16_qsu4c32s16s0_neon: each 32-bit lane holds K+0..3 in the
// low nibbles and K+4..7 in the high nibbles (XOR 0x88, scales pre-multiplied by 1/16).
// The SME2-only instructions are replaced as follows:
//   * luti4 (ZT0)              -> lsl #4 / and 0xF0 + sunpk + scvtf (FEAT_SME has no FEAT_LUT, so there is no LUT
//                                 argument).
//   * multi-vector ld1h/zip    -> per-vector ld1h with p3, p6 column predicates and zip1/zip2 (once per block).
//   * multi-vector mova/st1w   -> per-vector mova and st1w with p4, p1, p5, p7 column predicates.
//   * whilelt pn8/pn9          -> per-vector whilelt, computed once per N panel.
//   * fclamp                   -> fmin/fmax.

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
    KAI_ASM_CODE(matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa)
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
    mov z31.b, #-16                 // 0xF0 high-nibble mask
    ldr x16, [x0, #0x10]            // rhs_packed
    ldr x17, [x0, #0x18]            // rhs_scales
    ldr x1, [x0, #0x20]             // dst_stride_row
    ldr x6, [x0, #0x40]             // n
    ldr x3, [x0, #0x50]             // bl
    KAI_ASM_INST(0x8556c81c)        // ld1rw z28.s, p2/z, [x0, #0x58] // min
    KAI_ASM_INST(0x8557c81d)        // ld1rw z29.s, p2/z, [x0, #0x5c] // max

    mov x8, #0
    ldr x4, [x0]                    // dst
    cmp x8, x6
    b.ge n_end

    // N_LOOP START
KAI_ASM_LABEL(n_loop)
    // Column predicates of this N panel. They replace whilelt pn8.h (VLx2) and pn9.s (VLx4).
    whilelt p3.h, x8, x6            // RHS and scales, columns [0, 2 * cntw)
    whilelt p4.s, x8, x6            // dst, columns [0, cntw)
    add x2, x8, x14
    whilelt p1.s, x2, x6            // dst, columns [cntw, 2 * cntw)
    add x2, x2, x14
    whilelt p6.h, x2, x6            // RHS and scales, columns [2 * cntw, 4 * cntw)
    whilelt p5.s, x2, x6            // dst, columns [2 * cntw, 3 * cntw)
    add x2, x2, x14
    whilelt p7.s, x2, x6            // dst, columns [3 * cntw, 4 * cntw)
    ldr x9, [x0, #0x38]             // m
    ldr x22, [x0, #0x8]             // lhs_packed
    mov x24, x4                     // dst

    // M_LOOP START
KAI_ASM_LABEL(m_loop)
    KAI_ASM_INST(0xc00800ff)        // zero {za}
    KAI_ASM_INST(0x25a917e0)        // whilelt p0.s, xzr, x9
    mov x23, x22                    // LHS
    mov x20, x16                    // RHS
    mov x7, x17                     // RHS scales
    ldr x10, [x0, #0x48]            // k
    cbz x10, k_end

    // K_LOOP START: one block of bl values per iteration
KAI_ASM_LABEL(k_loop)
    ld1h z0.h, p3/z, [x7]
    ld1h z1.h, p6/z, [x7, #1, MUL VL]
    addvl x7, x7, #2
    zip1 z24.h, z0.h, z0.h
    zip2 z25.h, z0.h, z0.h
    zip1 z26.h, z1.h, z1.h
    zip2 z27.h, z1.h, z1.h
    mov x11, x3

    // BL_LOOP START: one 8-K packet of the s4s0sf16 layout per iteration
KAI_ASM_LABEL(block_loop)
    // Byte b of each column holds the low nibble K[b] and the high nibble K[b + 4].
    // ld2h splits bytes 0/1 (K0/K1, K4/K5) into z2/z12 and bytes 2/3 (K2/K3, K6/K7) into z3/z13.
    ld2h {z2.h, z3.h},   p3/z, [x20]
    ld2h {z12.h, z13.h}, p6/z, [x20, #2, MUL VL]
    addvl x20, x20, #4
    // LHS: one packet consumes 4 sequential K-pair groups z16 -> (K0, K1) .. z19 -> (K6, K7)
    ld1w z16.s, p0/z, [x23]
    ld1w z17.s, p0/z, [x23, #1, MUL VL]
    ld1w z18.s, p0/z, [x23, #2, MUL VL]
    ld1w z19.s, p0/z, [x23, #3, MUL VL]
    addvl x23, x23, #4
    // The top-justified nibble is 16 * value, and the packed scale is scale / 16.

    // K0/K1: low nibbles of bytes 0/1.
    lsl z20.b, z2.b, #4
    lsl z21.b, z12.b, #4
    sunpklo z8.h,  z20.b
    sunpkhi z9.h,  z20.b
    sunpklo z10.h, z21.b
    sunpkhi z11.h, z21.b
    scvtf z8.h,  p2/m, z8.h
    scvtf z9.h,  p2/m, z9.h
    scvtf z10.h, p2/m, z10.h
    scvtf z11.h, p2/m, z11.h
    fmul z8.h,  z8.h,  z24.h
    fmul z9.h,  z9.h,  z25.h
    fmul z10.h, z10.h, z26.h
    fmul z11.h, z11.h, z27.h
    KAI_ASM_INST(0x81a84a00)        // fmopa za0.s, p2/m, p2/m, z16.h, z8.h
    KAI_ASM_INST(0x81a94a01)        // fmopa za1.s, p2/m, p2/m, z16.h, z9.h
    KAI_ASM_INST(0x81aa4a02)        // fmopa za2.s, p2/m, p2/m, z16.h, z10.h
    KAI_ASM_INST(0x81ab4a03)        // fmopa za3.s, p2/m, p2/m, z16.h, z11.h

    // K2/K3: low nibbles of bytes 2/3.
    lsl z20.b, z3.b, #4
    lsl z21.b, z13.b, #4
    sunpklo z8.h,  z20.b
    sunpkhi z9.h,  z20.b
    sunpklo z10.h, z21.b
    sunpkhi z11.h, z21.b
    scvtf z8.h,  p2/m, z8.h
    scvtf z9.h,  p2/m, z9.h
    scvtf z10.h, p2/m, z10.h
    scvtf z11.h, p2/m, z11.h
    fmul z8.h,  z8.h,  z24.h
    fmul z9.h,  z9.h,  z25.h
    fmul z10.h, z10.h, z26.h
    fmul z11.h, z11.h, z27.h
    KAI_ASM_INST(0x81a84a20)        // fmopa za0.s, p2/m, p2/m, z17.h, z8.h
    KAI_ASM_INST(0x81a94a21)        // fmopa za1.s, p2/m, p2/m, z17.h, z9.h
    KAI_ASM_INST(0x81aa4a22)        // fmopa za2.s, p2/m, p2/m, z17.h, z10.h
    KAI_ASM_INST(0x81ab4a23)        // fmopa za3.s, p2/m, p2/m, z17.h, z11.h

    // K4/K5: high nibbles of bytes 0/1.
    and z20.d, z2.d, z31.d
    and z21.d, z12.d, z31.d
    sunpklo z8.h,  z20.b
    sunpkhi z9.h,  z20.b
    sunpklo z10.h, z21.b
    sunpkhi z11.h, z21.b
    scvtf z8.h,  p2/m, z8.h
    scvtf z9.h,  p2/m, z9.h
    scvtf z10.h, p2/m, z10.h
    scvtf z11.h, p2/m, z11.h
    fmul z8.h,  z8.h,  z24.h
    fmul z9.h,  z9.h,  z25.h
    fmul z10.h, z10.h, z26.h
    fmul z11.h, z11.h, z27.h
    KAI_ASM_INST(0x81a84a40)        // fmopa za0.s, p2/m, p2/m, z18.h, z8.h
    KAI_ASM_INST(0x81a94a41)        // fmopa za1.s, p2/m, p2/m, z18.h, z9.h
    KAI_ASM_INST(0x81aa4a42)        // fmopa za2.s, p2/m, p2/m, z18.h, z10.h
    KAI_ASM_INST(0x81ab4a43)        // fmopa za3.s, p2/m, p2/m, z18.h, z11.h

    // K6/K7: high nibbles of bytes 2/3.
    and z20.d, z3.d, z31.d
    and z21.d, z13.d, z31.d
    sunpklo z8.h,  z20.b
    sunpkhi z9.h,  z20.b
    sunpklo z10.h, z21.b
    sunpkhi z11.h, z21.b
    scvtf z8.h,  p2/m, z8.h
    scvtf z9.h,  p2/m, z9.h
    scvtf z10.h, p2/m, z10.h
    scvtf z11.h, p2/m, z11.h
    fmul z8.h,  z8.h,  z24.h
    fmul z9.h,  z9.h,  z25.h
    fmul z10.h, z10.h, z26.h
    fmul z11.h, z11.h, z27.h
    KAI_ASM_INST(0x81a84a60)        // fmopa za0.s, p2/m, p2/m, z19.h, z8.h
    KAI_ASM_INST(0x81a94a61)        // fmopa za1.s, p2/m, p2/m, z19.h, z9.h
    KAI_ASM_INST(0x81aa4a62)        // fmopa za2.s, p2/m, p2/m, z19.h, z10.h
    KAI_ASM_INST(0x81ab4a63)        // fmopa za3.s, p2/m, p2/m, z19.h, z11.h

    subs x11, x11, #8
    b.gt block_loop
    // BL_LOOP END

    subs x10, x10, x3
    b.gt k_loop
    // K_LOOP END

KAI_ASM_LABEL(k_end)
    cmp x9, x14
    csel x15, x9, x14, lo
    lsl x15, x15, #2
    mov w12, #0

    // STORE_LOOP START: one dst row per iteration
KAI_ASM_LABEL(store_loop)
    KAI_ASM_INST(0xc0020800)        // mova z0.b, p2/m, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020821)        // mova z1.b, p2/m, za0h.b[w12, 1]
    KAI_ASM_INST(0xc0020842)        // mova z2.b, p2/m, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020863)        // mova z3.b, p2/m, za0h.b[w12, 3]
    fmin z0.s, p2/m, z0.s, z29.s
    fmin z1.s, p2/m, z1.s, z29.s
    fmin z2.s, p2/m, z2.s, z29.s
    fmin z3.s, p2/m, z3.s, z29.s
    fmax z0.s, p2/m, z0.s, z28.s
    fmax z1.s, p2/m, z1.s, z28.s
    fmax z2.s, p2/m, z2.s, z28.s
    fmax z3.s, p2/m, z3.s, z28.s
    st1w z0.s, p4, [x24]
    st1w z1.s, p1, [x24, #1, MUL VL]
    st1w z2.s, p5, [x24, #2, MUL VL]
    st1w z3.s, p7, [x24, #3, MUL VL]
    add x24, x24, x1
    add w12, w12, #4
    cmp w12, w15
    b.lt store_loop
    // STORE_LOOP END

    ldr x5, [x0, #0x28]             // lhs_packed_stride
    add x22, x22, x5
    decw x9, all
    cmp x9, #0
    b.gt m_loop
    // M_LOOP END

    incb x4, all, mul #4            // dst
    ldr x5, [x0, #0x30]             // rhs_packed_stride
    add x16, x16, x5
    add x17, x17, x5
    incb x8, all
    cmp x8, x6
    b.lt n_loop
    // N_LOOP END

KAI_ASM_LABEL(n_end)
    KAI_ASM_INST(0xd503467f)        // smstop
    ldp     d14, d15, [sp, 96]
    ldp     d12, d13, [sp, 80]
    ldp     d10, d11, [sp, 64]
    ldp     d8, d9,   [sp, 48]
    ldp     x23, x24, [sp, 32]
    ldp     x21, x22, [sp, 16]
    ldp     x19, x20, [sp], 128
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f16p4vsx2_qsi4c32p16vsx4s4s0sf16_f16p_4vsx16vs_qmx_mopa)

    KAI_ASM_END
