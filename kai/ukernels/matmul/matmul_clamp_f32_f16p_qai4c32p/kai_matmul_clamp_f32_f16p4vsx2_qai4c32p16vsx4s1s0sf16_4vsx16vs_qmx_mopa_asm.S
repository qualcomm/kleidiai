//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

.macro ld1h_2, reg1, reg2, pred, addr
    ld1h \reg1, \pred/Z, [\addr]
    ld1h \reg2, \pred/Z, [\addr, #1, MUL VL]
.endm

.macro ld2b_2, reg1, reg2, pred, addr
    ld2b \reg1, \pred/Z, [\addr]
    ld2b \reg2, \pred/Z, [\addr, #1, MUL VL]
    addvl \addr, \addr, #2
.endm

.macro int4_to_fp16, o0, o1, o2, o3, src, tmp, pred
    mov     \tmp\().d, \src\().d
    lsr     \tmp\().b, \tmp\().b, #4      // tmp = high nibble
    and     \src\().b, \src\().b, #0x0f   // src = low nibble

    zip1    \o0\().b, \src\().b, \tmp\().b  // interleave low/high, first half of bytes
    zip2    \o2\().b, \src\().b, \tmp\().b  // interleave low/high, second half of bytes

    uunpkhi \o1\().h, \o0\().b               // widen: upper half -> o1
    uunpklo \o0\().h, \o0\().b               // widen: lower half -> o0 (must widen o1 first!)
    uunpkhi \o3\().h, \o2\().b
    uunpklo \o2\().h, \o2\().b

    ucvtf   \o0\().h, \pred/m, \o0\().h
    ucvtf   \o1\().h, \pred/m, \o1\().h
    ucvtf   \o2\().h, \pred/m, \o2\().h
    ucvtf   \o3\().h, \pred/m, \o3\().h
.endm

.macro move_tile_vector,  reg1, reg2, reg3, reg4,tile, addr, pred
    mova \reg1, \pred/M, \tile[\addr, 0] 
    mova \reg2, \pred/M, \tile[\addr, 1] 
    mova \reg3, \pred/M, \tile[\addr, 2] 
    mova \reg4, \pred/M, \tile[\addr, 3] 
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
    ldr x19, [x0, #0x50]             // lut
    KAI_ASM_INST(0xe11f8260)         // ldr zt0, [x19]
    mov x8, #0
    ldr x16, [x0, #0x10]             // rhs_packed
    ldr x6, [x0, #0x38]              // n
    ldr x21, [x0, #0x18]             // dst_stride_row
    ldr x23, [x0, #0x40]             // k
    KAI_ASM_INST(0x8556c81c)         // ld1rw z28.s, p2/z, [x0, #0x58] // min
    KAI_ASM_INST(0x8557c81d)         // ld1rw z29.s, p2/z, [x0, #0x5c] // max
    // mov x25, #0                      // DEBUG: one-shot flags (bit0 = block dbg, bit1 = store dbg)
    mov x25, #0                      // DEBUG: one-shot flags (bit0 = block dbg, bit1 = store dbg)

    ptrue p7.h
    // KAI_ASM_INST(0x25a66511)         // whilelt pn9.s, x8, x6, VLx4
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
    ld1h_2 z0.h, z1.h, p7, x17       // KAI_ASM_INST(0xa0402220)         // ld1h { z0.h - z1.h }, pn8/z, [x17] // rhs offset, packed in q4k nrx4 order
	addvl x1, x17, #2
    ld1h_2 z2.h, z3.h, p7, x1       // KAI_ASM_INST(0xa0412222)         // ld1h { z2.h - z3.h }, pn8/z, [x17, #0x2, mul vl] // rhs scale, packed in q4k nrx4 order
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
    KAI_ASM_INST(0xa540a066)                        // ld1w {z6.s}, p0/z, [x3]
    int4_to_fp16 z8, z9, z10, z11, z4, z30, p2     // KAI_ASM_INST(0xc08a9088)         // luti4 {z8.h - z11.h}, zt0, z4[0]
    mov     z16.d, z24.d
    mov     z17.d, z25.d
    mov     z18.d, z26.d
    mov     z19.d, z27.d
    fmla    z16.h, p2/m, z8.h, z0.h
    fmla    z17.h, p2/m, z9.h, z1.h
    fmla    z18.h, p2/m, z10.h, z2.h
    fmla    z19.h, p2/m, z11.h, z3.h
    // tbnz    x25, #0, label_dbg_block_skip   // DEBUG: only stop on the very first block iteration
    // orr     x25, x25, #1
    // KAI_ASM_INST(0xd4200000)         // DEBUG: brk #0 -- inspect z4/z5(raw), z8-z11(dequant), z6(lhs), z16-z19(scaled), z0-z3/z24-z27
// KAI_ASM_LABEL(label_dbg_block_skip)
    KAI_ASM_INST(0x81b048c0)         // fmopa za0.s, p2/m, p2/m, z6.h, z16.h
    KAI_ASM_INST(0x81b148c1)         // fmopa za1.s, p2/m, p2/m, z6.h, z17.h
    KAI_ASM_INST(0x81b248c2)         // fmopa za2.s, p2/m, p2/m, z6.h, z18.h
    KAI_ASM_INST(0x81b348c3)         // fmopa za3.s, p2/m, p2/m, z6.h, z19.h
    KAI_ASM_INST(0xa54e4067)         // ld1w {z7.s}, p0/z, [x3, x14, lsl #2]
    add x3, x3, x14, lsl #3
    int4_to_fp16 z12, z13, z14, z15, z5, z30, p2   // KAI_ASM_INST(0xc08a90ac)         // luti4 {z12.h - z15.h}, zt0, z5[0]
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
    // tbnz    x25, #1, label_dbg_store_skip   // DEBUG: only stop on the very first store iteration
    // orr     x25, x25, #2
    // KAI_ASM_INST(0xd4200000)         // DEBUG: brk #0 -- inspect za0h-za3h[w12] before extraction/clamp/store
// KAI_ASM_LABEL(label_dbg_store_skip)
    move_tile_vector z0.b, z1.b, z2.b, z3.b, za0h.b, w12, p2      // KAI_ASM_INST(0xc0060400)         // mova {z0.b - z3.b}, za0h.b[w12, 0:3]
    clamp_float_4 z0.s, z1.s, z2.s, z3.s, z28.s, z29.s,p2         //KAI_ASM_INST(0xc1bdcb80)         // fclamp { z0.s - z3.s }, z28.s, z29.s
    //KAI_ASM_INST(0xa060c700)         // st1w {z0.s - z3.s}, pn9, [x24]
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
    //KAI_ASM_INST(0x25a66511)         // whilelt pn9.s, x8, x6, VLx4
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
