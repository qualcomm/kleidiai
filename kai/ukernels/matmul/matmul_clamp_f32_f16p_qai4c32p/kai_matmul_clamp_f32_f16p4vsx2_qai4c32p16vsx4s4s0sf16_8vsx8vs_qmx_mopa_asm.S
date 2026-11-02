//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
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
    KAI_ASM_CODE(matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s4s0sf16_8vsx8vs_qmx_mopa)
    KAI_ASM_ALIGN
    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s4s0sf16_8vsx8vs_qmx_mopa)

// 2VLx2VL ZA tiling: each RHS panel (16vs columns) is processed in two 2VL column halves.
// For one half, za0/za1 hold LHS block A (rows 0..VL-1) and za2/za3 hold LHS block B
// (rows VL..2VL-1), so every decoded RHS vector feeds two FMOPAs instead of one.
// Packed LHS/RHS layouts are unchanged.
KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s4s0sf16_8vsx8vs_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s4s0sf16_8vsx8vs_qmx_mopa)
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
    mov z31.b, #-16
    mov x8, #0
    ldr x16, [x0, #0x10]             // rhs_packed
    ldr x6, [x0, #0x38]              // n
    ldr x21, [x0, #0x18]             // dst_stride_row
    ldr x20, [x0, #0x20]             // lhs_packed_stride
    ldr x23, [x0, #0x40]             // k
    KAI_ASM_INST(0x8556c81c)         // ld1rw z28.s, p2/z, [x0, #0x58] // min
    KAI_ASM_INST(0x8557c81d)         // ld1rw z29.s, p2/z, [x0, #0x5c] // max
    ldr x4, [x0]                     // dst
    cmp x8, x6
    b.ge label_9
KAI_ASM_LABEL(label_1)               // N loop (16vs-column RHS panel)
    mov x13, #0                      // column half index
    pfalse p1.b                      // half selector for metadata: false -> first half
    mov x19, x8                      // first column of this half
    mov x7, #0                       // RHS payload offset of this half, in halfwords
    mov x25, x4                      // dst base of this half
KAI_ASM_LABEL(label_11)              // column-half loop
    mov x1, x19
    whilelt p3.s, x1, x6             // store predicates for the two VL column groups of this half
    incw x1
    whilelt p4.s, x1, x6
    ldr x9, [x0, #0x30]              // m
    ldr x22, [x0, #0x8]              // lhs_packed
    mov x24, x25                     // dst tile base
KAI_ASM_LABEL(label_2)               // M loop (two LHS blocks per iteration)
    KAI_ASM_INST(0xc00800ff)         // zero {za}
    KAI_ASM_INST(0x25a917e0)         // whilelt p0.s, xzr, x9
    whilelt p7.s, x14, x9            // rows of the second LHS block; inactive lanes are not accessed
    mov x26, x16                     // rhs block pointer
    mov x3, x22                      // lhs block A pointer
    add x2, x22, x20                 // lhs block B pointer
    mov x10, x23                     // k remaining
    cmp x10, #0
    b.eq label_8
KAI_ASM_LABEL(label_3)               // K loop
    ldr x11, [x0, #0x48]             // bl
    addvl x17, x26, #16              // rhs metadata
    // Metadata is interleaved [half0 row i, half1 row i]; ld2h splits it per column half.
    ld2h {z12.h, z13.h}, p2/z, [x17]         // rhs offsets
    addvl x1, x17, #2
    ld2h {z22.h, z23.h}, p2/z, [x1]          // rhs scales
    sel z12.h, p1, z13.h, z12.h
    sel z22.h, p1, z23.h, z22.h
    zip1 z24.h, z12.h, z12.h         // duplicate offsets for the K-pair layout
    zip2 z25.h, z12.h, z12.h
    zip1 z0.h, z22.h, z22.h          // duplicate scales for the K-pair layout
    zip2 z1.h, z22.h, z22.h
KAI_ASM_LABEL(label_4)               // block loop
    // Payload byte b holds signed (q - 8): low nibble K[b], high nibble K[b + 4].
    // ld2h maps consecutive row-pairs of this half to the two ZA column groups.
    ld2h {z4.h, z5.h}, p2/z, [x26, x7, lsl #1]
    addvl x26, x26, #4
    // One S4S0 packet consumes four consecutive packed-LHS K-pair groups per LHS block.
    ld1w z6.s,  p0/z, [x3]
    ld1w z7.s,  p0/z, [x3, #1, mul vl]
    ld1w z14.s, p0/z, [x3, #2, mul vl]
    ld1w z15.s, p0/z, [x3, #3, mul vl]
    addvl x3, x3, #4
    ld1w z16.s, p7/z, [x2]
    ld1w z17.s, p7/z, [x2, #1, mul vl]
    ld1w z18.s, p7/z, [x2, #2, mul vl]
    ld1w z19.s, p7/z, [x2, #3, mul vl]
    addvl x2, x2, #4

    // K0/K1: low nibbles of bytes 0/1.
    lsl z20.b, z4.b, #4
    sunpklo z8.h, z20.b
    sunpkhi z9.h, z20.b
    scvtf z8.h, p2/m, z8.h
    scvtf z9.h, p2/m, z9.h
    // fmad: zdn = za + zdn * zm -> value*scale + offset in place
    fmad z8.h, p2/m, z0.h, z24.h
    fmad z9.h, p2/m, z1.h, z25.h
    // K2/K3: low nibbles of bytes 2/3, decoded while the K0/K1 outer products issue.
    lsl z21.b, z5.b, #4
    KAI_ASM_INST(0x81a848c0)         // fmopa za0.s, p2/m, p2/m, z6.h, z8.h
    KAI_ASM_INST(0x81a948c1)         // fmopa za1.s, p2/m, p2/m, z6.h, z9.h
    KAI_ASM_INST(0x81a84a02)         // fmopa za2.s, p2/m, p2/m, z16.h, z8.h
    KAI_ASM_INST(0x81a94a03)         // fmopa za3.s, p2/m, p2/m, z16.h, z9.h
    sunpklo z10.h, z21.b
    sunpkhi z11.h, z21.b
    scvtf z10.h, p2/m, z10.h
    scvtf z11.h, p2/m, z11.h
    fmad z10.h, p2/m, z0.h, z24.h
    fmad z11.h, p2/m, z1.h, z25.h
    // K4/K5: high nibbles of bytes 0/1.
    and z20.d, z4.d, z31.d
    KAI_ASM_INST(0x81aa48e0)         // fmopa za0.s, p2/m, p2/m, z7.h, z10.h
    KAI_ASM_INST(0x81ab48e1)         // fmopa za1.s, p2/m, p2/m, z7.h, z11.h
    KAI_ASM_INST(0x81aa4a22)         // fmopa za2.s, p2/m, p2/m, z17.h, z10.h
    KAI_ASM_INST(0x81ab4a23)         // fmopa za3.s, p2/m, p2/m, z17.h, z11.h
    sunpklo z8.h, z20.b
    sunpkhi z9.h, z20.b
    scvtf z8.h, p2/m, z8.h
    scvtf z9.h, p2/m, z9.h
    fmad z8.h, p2/m, z0.h, z24.h
    fmad z9.h, p2/m, z1.h, z25.h
    // K6/K7: high nibbles of bytes 2/3.
    and z21.d, z5.d, z31.d
    KAI_ASM_INST(0x81a849c0)         // fmopa za0.s, p2/m, p2/m, z14.h, z8.h
    KAI_ASM_INST(0x81a949c1)         // fmopa za1.s, p2/m, p2/m, z14.h, z9.h
    KAI_ASM_INST(0x81a84a42)         // fmopa za2.s, p2/m, p2/m, z18.h, z8.h
    KAI_ASM_INST(0x81a94a43)         // fmopa za3.s, p2/m, p2/m, z18.h, z9.h
    sunpklo z10.h, z21.b
    sunpkhi z11.h, z21.b
    scvtf z10.h, p2/m, z10.h
    scvtf z11.h, p2/m, z11.h
    fmad z10.h, p2/m, z0.h, z24.h
    fmad z11.h, p2/m, z1.h, z25.h
    KAI_ASM_INST(0x81aa49e0)         // fmopa za0.s, p2/m, p2/m, z15.h, z10.h
    KAI_ASM_INST(0x81ab49e1)         // fmopa za1.s, p2/m, p2/m, z15.h, z11.h
    KAI_ASM_INST(0x81aa4a62)         // fmopa za2.s, p2/m, p2/m, z19.h, z10.h
    KAI_ASM_INST(0x81ab4a63)         // fmopa za3.s, p2/m, p2/m, z19.h, z11.h

    subs x11, x11, #8
    b.gt label_4
    addvl x26, x17, #4               // next rhs block - skipping metadata
    ldr x11, [x0, #0x48]             // bl
    subs x10, x10, x11
    b.gt label_3
KAI_ASM_LABEL(label_8)               // store LHS block A rows (za0, za1)
    cmp x9, x14
    csel x15, x9, x14, lo
    lsl x15, x15, #2
    mov w12, #0
KAI_ASM_LABEL(label_5)
    KAI_ASM_INST(0xc0020800)         // mova z0.b, p2/M, za0h.b[w12, 0]
    KAI_ASM_INST(0xc0020821)         // mova z1.b, p2/M, za0h.b[w12, 1]
    fmin z0.s, p2/m, z0.s, z29.s
    fmin z1.s, p2/m, z1.s, z29.s
    fmax z0.s, p2/m, z0.s, z28.s
    fmax z1.s, p2/m, z1.s, z28.s
    st1w {z0.s}, p3, [x24]
    st1w {z1.s}, p4, [x24, #1, MUL VL]
    add x24, x24, x21
    add w12, w12, #4
    cmp w12, w15
    blt label_5
    // store LHS block B rows (za2, za3), if any
    subs x15, x9, x14
    b.le label_10
    cmp x15, x14
    csel x15, x15, x14, lo
    lsl x15, x15, #2
    mov w12, #0
KAI_ASM_LABEL(label_6)
    KAI_ASM_INST(0xc0020842)         // mova z2.b, p2/M, za0h.b[w12, 2]
    KAI_ASM_INST(0xc0020863)         // mova z3.b, p2/M, za0h.b[w12, 3]
    fmin z2.s, p2/m, z2.s, z29.s
    fmin z3.s, p2/m, z3.s, z29.s
    fmax z2.s, p2/m, z2.s, z28.s
    fmax z3.s, p2/m, z3.s, z28.s
    st1w {z2.s}, p3, [x24]
    st1w {z3.s}, p4, [x24, #1, MUL VL]
    add x24, x24, x21
    add w12, w12, #4
    cmp w12, w15
    blt label_6
KAI_ASM_LABEL(label_10)              // next M tile pair
    add x22, x22, x20, lsl #1
    sub x9, x9, x14, lsl #1
    cmp x9, #0
    b.gt label_2
    // next column half of this panel
    add x13, x13, #1
    cmp x13, #2
    b.ge label_12
    add x19, x8, x14, lsl #1         // second half starts 2VL columns in
    cmp x19, x6
    b.ge label_12                    // no columns left in the second half
    ptrue p1.h                       // half selector: true -> second half
    cnth x7, all, mul #2             // payload offset of the second half (2VL bytes), in halfwords
    addvl x25, x4, #2                // dst of the second half
    b label_11
KAI_ASM_LABEL(label_12)              // next N panel
    incb x4, all, mul #4             // dst
    ldr x5, [x0, #0x28]              // rhs_packed_stride
    add x16, x16, x5
    incb x8, all
    cmp x8, x6
    b.lt label_1
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
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s4s0sf16_8vsx8vs_qmx_mopa)

    KAI_ASM_END
