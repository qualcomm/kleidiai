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

    KAI_ASM_CODE(matmul_clamp_f16_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_f16_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_f16_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_f16_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa)
    stp     x19, x20, [sp, -112]!
    stp     x21, x28, [sp, 16]
    stp     d8, d9,   [sp, 32]
    stp     d10, d11, [sp, 48]
    stp     d12, d13, [sp, 64]
    stp     d14, d15, [sp, 80]

    /* ---------------------------------------------------
                  Registers allocations
        x6:  RHS ragged-tail byte-block size (rhs_row_bytes)
        x7:  SME vector length in words (cntw)
        x8:  RHS base address for this M-block (rhs)
        x9:  Destination base address for this M-block (dst)
        x10: LHS pointer (lhs)
        x11: RHS pointer (rhs)
        x12: Remaining M elements
        x13: Remaining N elements
        x14: k exit condition (k_cond) / ZA tile index (l_idx)
        x15: LHS scaling factor pointer (lhs_sf_ptr)
        x16: ZA tile exit condition (l_cnd)
        x17: Destination pointer (dst)
        x19: Destination outer address (dst)
        x20: LHS base address (lhs)
        x21: N-remaining predicate counter (self-cancelling per N-tile)
        x28: RHS ragged-tail byte boundary
        z28: 0xf0 mask, used to isolate the high nibble (already pre-scaled by 16 in its byte position)
        z30: clamp_min (fp32, broadcast)
        z31: clamp_max (fp32, broadcast)
    --------------------------------------------------- */

    KAI_ASM_INST(0xd503477f)             // smstart
    ldr     x19, [x0, #0x00]             // dst
    ldr     x20, [x0, #0x08]             // lhs_packed
    ldr     x6,  [x0, #0x50]             // rhs_row_bytes
    cntw    x7
    mov     w9, #0xf0
    dup     z28.b, w9
    ptrue   p2.b
    ld1rw   {z30.s}, p2/Z, [x0, #0x58]   // clamp_min
    ld1rw   {z31.s}, p2/Z, [x0, #0x5c]   // clamp_max
    ldr     x12, [x0, #0x20]             // m

    whilelt p0.s, xzr, x12
KAI_ASM_LABEL(label_1)
    mov     x8, x8
    ldr     x8,  [x0, #0x10]             // rhs_packed
    add     x28, x8, x6
    mov     x9, x19
    ldr     x13, [x0, #0x28]             // n
    cmp     x7, x12
    csel    x16, x7, x12, lt
    lsl     x16, x16, #2
    mov     x21, #0
KAI_ASM_LABEL(label_2)
    mov     x10, x20
    mov     x11, x8
    mov     x17, x9
    KAI_ASM_INST(0xc00800ff)             // zero {za}
    ldr     x14, [x0, #0x40]             // m_blk
    add     x14, x10, x14
KAI_ASM_LABEL(label_3)
    // LHS: two groups of kai_kr=4 int8 K-values, 16 elements (one VL of int8 packed as int32) apart,
    // matching the RHS nibble-pairing distance produced by kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.
    ld1w    { z16.s }, p0/z, [x10]
    ld1w    { z20.s }, p0/z, [x10, #0x4, mul vl]
    // RHS: each of the kai_nr=4 output lanes owns its own separate byte stream (no two lanes share a
    // loaded register), matching the per-lane layout produced by kai_rhs_pack_nxk_qsi4cxp_qs4cxs1s0.
    mov     x15, x11
    whilelt p3.h, x15, x28
    addvl   x15, x11, #1
    whilelt p4.h, x15, x28
    addvl   x15, x11, #2
    whilelt p6.h, x15, x28
    addvl   x15, x11, #3
    whilelt p7.h, x15, x28
    ld1h    z0.h, p3/z, [x11]
    ld1h    z1.h, p4/z, [x11, #1, mul vl]
    ld1h    z2.h, p6/z, [x11, #2, mul vl]
    ld1h    z3.h, p7/z, [x11, #3, mul vl]
    // Low nibble: shift left by 4 so the value lands pre-scaled by 16 in its natural sign position,
    // ready to feed smopa directly - no sign-extension or lane reordering needed.
    lsl     z24.b, z0.b, #4
    lsl     z25.b, z1.b, #4
    lsl     z26.b, z2.b, #4
    lsl     z27.b, z3.b, #4
    KAI_ASM_INST(0xa0984a00)             // smopa   za0.s, p2/m, p2/m, z16.b, z24.b
    KAI_ASM_INST(0xa0994a01)             // smopa   za1.s, p2/m, p2/m, z16.b, z25.b
    KAI_ASM_INST(0xa09a4a02)             // smopa   za2.s, p2/m, p2/m, z16.b, z26.b
    KAI_ASM_INST(0xa09b4a03)             // smopa   za3.s, p2/m, p2/m, z16.b, z27.b
    // High nibble: mask off the low nibble with 0xf0. This nibble packs the K-value 16 positions
    // ahead, already pre-scaled by 16 in its byte position, so no shift/zip is required here.
    and     z0.d, z0.d, z28.d
    and     z1.d, z1.d, z28.d
    and     z2.d, z2.d, z28.d
    and     z3.d, z3.d, z28.d
    KAI_ASM_INST(0xa0804a80)             // smopa   za0.s, p2/m, p2/m, z20.b, z0.b
    KAI_ASM_INST(0xa0814a81)             // smopa   za1.s, p2/m, p2/m, z20.b, z1.b
    KAI_ASM_INST(0xa0824a82)             // smopa   za2.s, p2/m, p2/m, z20.b, z2.b
    KAI_ASM_INST(0xa0834a83)             // smopa   za3.s, p2/m, p2/m, z20.b, z3.b
    ld1w    { z17.s }, p0/z, [x10, #0x1, mul vl]
    ld1w    { z21.s }, p0/z, [x10, #0x5, mul vl]
    addvl   x15, x11, #4
    whilelt p3.h, x15, x28
    addvl   x15, x11, #5
    whilelt p4.h, x15, x28
    addvl   x15, x11, #6
    whilelt p6.h, x15, x28
    addvl   x15, x11, #7
    whilelt p7.h, x15, x28
    ld1h    z4.h, p3/z, [x11, #4, mul vl]
    ld1h    z5.h, p4/z, [x11, #5, mul vl]
    ld1h    z6.h, p6/z, [x11, #6, mul vl]
    ld1h    z7.h, p7/z, [x11, #7, mul vl]
    addvl   x11, x11, #8
    lsl     z24.b, z4.b, #4
    lsl     z25.b, z5.b, #4
    lsl     z26.b, z6.b, #4
    lsl     z27.b, z7.b, #4
    KAI_ASM_INST(0xa0984a20)             // smopa   za0.s, p2/m, p2/m, z17.b, z24.b
    KAI_ASM_INST(0xa0994a21)             // smopa   za1.s, p2/m, p2/m, z17.b, z25.b
    KAI_ASM_INST(0xa09a4a22)             // smopa   za2.s, p2/m, p2/m, z17.b, z26.b
    KAI_ASM_INST(0xa09b4a23)             // smopa   za3.s, p2/m, p2/m, z17.b, z27.b
    and     z4.d, z4.d, z28.d
    and     z5.d, z5.d, z28.d
    and     z6.d, z6.d, z28.d
    and     z7.d, z7.d, z28.d
    KAI_ASM_INST(0xa0844aa0)             // smopa   za0.s, p2/m, p2/m, z21.b, z4.b
    KAI_ASM_INST(0xa0854aa1)             // smopa   za1.s, p2/m, p2/m, z21.b, z5.b
    KAI_ASM_INST(0xa0864aa2)             // smopa   za2.s, p2/m, p2/m, z21.b, z6.b
    KAI_ASM_INST(0xa0874aa3)             // smopa   za3.s, p2/m, p2/m, z21.b, z7.b
    ld1w    { z18.s }, p0/z, [x10, #0x2, mul vl]
    ld1w    { z22.s }, p0/z, [x10, #0x6, mul vl]
    mov     x15, x11
    whilelt p3.h, x15, x28
    addvl   x15, x11, #1
    whilelt p4.h, x15, x28
    addvl   x15, x11, #2
    whilelt p6.h, x15, x28
    addvl   x15, x11, #3
    whilelt p7.h, x15, x28
    ld1h    z8.h,  p3/z, [x11]
    ld1h    z9.h,  p4/z, [x11, #1, mul vl]
    ld1h    z10.h, p6/z, [x11, #2, mul vl]
    ld1h    z11.h, p7/z, [x11, #3, mul vl]
    lsl     z24.b, z8.b, #4
    lsl     z25.b, z9.b, #4
    lsl     z26.b, z10.b, #4
    lsl     z27.b, z11.b, #4
    KAI_ASM_INST(0xa0984a40)             // smopa   za0.s, p2/m, p2/m, z18.b, z24.b
    KAI_ASM_INST(0xa0994a41)             // smopa   za1.s, p2/m, p2/m, z18.b, z25.b
    KAI_ASM_INST(0xa09a4a42)             // smopa   za2.s, p2/m, p2/m, z18.b, z26.b
    KAI_ASM_INST(0xa09b4a43)             // smopa   za3.s, p2/m, p2/m, z18.b, z27.b
    and     z8.d,  z8.d,  z28.d
    and     z9.d,  z9.d,  z28.d
    and     z10.d, z10.d, z28.d
    and     z11.d, z11.d, z28.d
    KAI_ASM_INST(0xa0884ac0)             // smopa   za0.s, p2/m, p2/m, z22.b, z8.b
    KAI_ASM_INST(0xa0894ac1)             // smopa   za1.s, p2/m, p2/m, z22.b, z9.b
    KAI_ASM_INST(0xa08a4ac2)             // smopa   za2.s, p2/m, p2/m, z22.b, z10.b
    KAI_ASM_INST(0xa08b4ac3)             // smopa   za3.s, p2/m, p2/m, z22.b, z11.b
    ld1w    { z19.s }, p0/z, [x10, #0x3, mul vl]
    ld1w    { z23.s }, p0/z, [x10, #0x7, mul vl]
    addvl   x15, x11, #4
    whilelt p3.h, x15, x28
    addvl   x15, x11, #5
    whilelt p4.h, x15, x28
    addvl   x15, x11, #6
    whilelt p6.h, x15, x28
    addvl   x15, x11, #7
    whilelt p7.h, x15, x28
    ld1h    z12.h, p3/z, [x11, #4, mul vl]
    ld1h    z13.h, p4/z, [x11, #5, mul vl]
    ld1h    z14.h, p6/z, [x11, #6, mul vl]
    ld1h    z15.h, p7/z, [x11, #7, mul vl]
    lsl     z24.b, z12.b, #4
    lsl     z25.b, z13.b, #4
    lsl     z26.b, z14.b, #4
    lsl     z27.b, z15.b, #4
    KAI_ASM_INST(0xa0984a60)             // smopa   za0.s, p2/m, p2/m, z19.b, z24.b
    KAI_ASM_INST(0xa0994a61)             // smopa   za1.s, p2/m, p2/m, z19.b, z25.b
    KAI_ASM_INST(0xa09a4a62)             // smopa   za2.s, p2/m, p2/m, z19.b, z26.b
    KAI_ASM_INST(0xa09b4a63)             // smopa   za3.s, p2/m, p2/m, z19.b, z27.b
    and     z12.d, z12.d, z28.d
    and     z13.d, z13.d, z28.d
    and     z14.d, z14.d, z28.d
    and     z15.d, z15.d, z28.d
    KAI_ASM_INST(0xa08c4ae0)             // smopa   za0.s, p2/m, p2/m, z23.b, z12.b
    KAI_ASM_INST(0xa08d4ae1)             // smopa   za1.s, p2/m, p2/m, z23.b, z13.b
    KAI_ASM_INST(0xa08e4ae2)             // smopa   za2.s, p2/m, p2/m, z23.b, z14.b
    KAI_ASM_INST(0xa08f4ae3)             // smopa   za3.s, p2/m, p2/m, z23.b, z15.b
    addvl   x11, x11, #8
    addvl   x10, x10, #8
    cmp     x10, x14
    b.lt    label_3

    // N-remaining predicates for the fp32 row-sum/scale/bias registers and the fp32-domain clamp.
    whilelt p4.s, x21, x13
    incw    x21
    whilelt p5.s, x21, x13
    incw    x21
    whilelt p6.s, x21, x13
    incw    x21
    whilelt p7.s, x21, x13
    decw    x21
    decw    x21
    decw    x21
    // N-remaining predicates for the packed fp16 store: register #1 covers output columns [0, 2*VL),
    // register #2 covers columns [2*VL, 4*VL) - matching the two-register uzp1-packed store layout below.
    // st1h only accepts a governing predicate in p0-p7, so reuse p1 (freshly overwritten by the N-tail
    // whilelt right after this loop) and p3 (dead once the K-loop's ragged-tail checks are done).
    whilelt p1.h, x21, x13
    inch    x21
    whilelt p3.h, x21, x13
    dech    x21

    // Load RHS row sum, scale factors, and bias.
    ld1w z0.s, p4/Z, [x11]
    ld1w z1.s, p5/Z, [x11, #1, mul vl]
    ld1w z2.s, p6/Z, [x11, #2, mul vl]
    ld1w z3.s, p7/Z, [x11, #3, mul vl]
    addvl x11, x11, #4
    ld1w z4.s, p4/Z, [x11]
    ld1w z5.s, p5/Z, [x11, #1, mul vl]
    ld1w z6.s, p6/Z, [x11, #2, mul vl]
    ld1w z7.s, p7/Z, [x11, #3, mul vl]
    addvl x11, x11, #4
    ld1w z8.s, p4/Z, [x11]
    ld1w z9.s, p5/Z, [x11, #1, mul vl]
    ld1w z10.s, p6/Z, [x11, #2, mul vl]
    ld1w z11.s, p7/Z, [x11, #3, mul vl]
    addvl x11, x11, #4

    mov     x14, #0
    addvl   x15, x10, #1
KAI_ASM_LABEL(label_4)
    ld1rw   {z16.s}, p2/z, [x10]
    ld1rw   {z17.s}, p2/z, [x15]
    add     x10, x10, #4
    add     x15, x15, #4

    fmul    z20.s, z17.s, z4.s
    fmul    z21.s, z17.s, z5.s
    fmul    z22.s, z17.s, z6.s
    fmul    z23.s, z17.s, z7.s

    KAI_ASM_INST(0xc002480c)             // mova z12.b, p2/M, za0h.b[w14, 0]
    KAI_ASM_INST(0xc002482d)             // mova z13.b, p2/M, za0h.b[w14, 1]
    KAI_ASM_INST(0xc002484e)             // mova z14.b, p2/M, za0h.b[w14, 2]
    KAI_ASM_INST(0xc002486f)             // mova z15.b, p2/M, za0h.b[w14, 3]

    mla     z12.s, p2/m, z16.s, z0.s
    mla     z13.s, p2/m, z16.s, z1.s
    mla     z14.s, p2/m, z16.s, z2.s
    mla     z15.s, p2/m, z16.s, z3.s

    scvtf   z12.s, p2/M, z12.s
    scvtf   z13.s, p2/M, z13.s
    scvtf   z14.s, p2/M, z14.s
    scvtf   z15.s, p2/M, z15.s

    fmul    z24.s, z12.s, z20.s
    fmul    z25.s, z13.s, z21.s
    fmul    z26.s, z14.s, z22.s
    fmul    z27.s, z15.s, z23.s

    fadd    z24.s, p2/m, z24.s, z8.s
    fadd    z25.s, p2/m, z25.s, z9.s
    fadd    z26.s, p2/m, z26.s, z10.s
    fadd    z27.s, p2/m, z27.s, z11.s

    fmin    z24.s, p2/M, z24.s, z31.s
    fmin    z25.s, p2/M, z25.s, z31.s
    fmin    z26.s, p2/M, z26.s, z31.s
    fmin    z27.s, p2/M, z27.s, z31.s
    fmax    z24.s, p2/M, z24.s, z30.s
    fmax    z25.s, p2/M, z25.s, z30.s
    fmax    z26.s, p2/M, z26.s, z30.s
    fmax    z27.s, p2/M, z27.s, z30.s

    // Pack two adjacent fp32 lanes into one fully-packed fp16 vector: FCVT (S->H) places each converted
    // value at the even 16-bit slot of its 32-bit source lane; UZP1 collects the even slots of both
    // sources, producing lane0's values followed by lane1's values - i.e. plain concatenation.
    fcvt    z24.h, p2/m, z24.s
    fcvt    z16.h, p2/m, z25.s
    fcvt    z26.h, p2/m, z26.s
    fcvt    z17.h, p2/m, z27.s
    uzp1    z16.h, z24.h, z16.h
    uzp1    z17.h, z26.h, z17.h

    st1h    z16.h, p1, [x17]
    st1h    z17.h, p3, [x17, #1, mul vl]

    ldr     x5, [x0, #0x18]              // dst_stride_row
    add     x17, x17, x5
    add     x14, x14, #4
    cmp     x14, x16
    b.lt    label_4

    ldr     x14, [x0, #0x38]             // rhs_stride
    add     x8, x8, x14
    addvl   x9, x9, #2
    decb    x13, all
    add     x28, x8, x6
    whilelt p1.h, xzr, x13
    b.mi    label_2

    ldr     x14, [x0, #0x30]             // lhs_stride
    add     x20, x20, x14
    ldr     x14, [x0, #0x48]             // dst_inc
    add     x19, x19, x14
    sub     x12, x12, x7
    whilelt p0.s, xzr, x12
    b.mi    label_1

    KAI_ASM_INST(0xd503467f)             // smstop

    ldp     d14, d15, [sp, 80]
    ldp     d12, d13, [sp, 64]
    ldp     d10, d11, [sp, 48]
    ldp     d8, d9,   [sp, 32]
    ldp     x21, x28, [sp, 16]
    ldp     x19, x20, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_f16_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_qmx_mopa)

    KAI_ASM_END
