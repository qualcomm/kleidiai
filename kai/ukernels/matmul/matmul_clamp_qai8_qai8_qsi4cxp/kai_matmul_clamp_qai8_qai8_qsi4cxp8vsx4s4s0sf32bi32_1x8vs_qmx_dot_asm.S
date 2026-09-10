//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//

#if defined(_MSC_VER) && !defined(__clang__)
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

    KAI_ASM_CODE(matmul_clamp_qai8_qai8_qsi4cxp8vsx4s4s0sf32bi32_1x8vs_qmx_dot)
    KAI_ASM_ALIGN

    KAI_ASM_GLOBAL(kai_kernel_matmul_clamp_qai8_qai8_qsi4cxp8vsx4s4s0sf32bi32_1x8vs_qmx_dot)

KAI_ASM_FUNCTION_TYPE(kai_kernel_matmul_clamp_qai8_qai8_qsi4cxp8vsx4s4s0sf32bi32_1x8vs_qmx_dot)
KAI_ASM_FUNCTION_LABEL(kai_kernel_matmul_clamp_qai8_qai8_qsi4cxp8vsx4s4s0sf32bi32_1x8vs_qmx_dot)
    // QMX (SME1) GEMV over the "s4s0" i4 RHS container -- the same packed buffer the
    // 8vsx8vs_qmx_mopa GEMM kernel consumes. No zt0/luti4: the i4 -> i8 expansion is arithmetic,
    // so args.lut_ptr is unused and left NULL.
    //
    // Layout recap (one nr-column panel, nr = 8*vscale, VL = nr*2 bytes):
    //   [ nr*4 bias | payload: roundup(K,32)*nr/2 | nr*4 scale ]
    // A 32-K block is 8 payload vectors. Vector kg8*2 + n_half holds nr/2 columns; column j sits at
    // bytes 4j..4j+3, and byte 4j+i packs K[32b + 8*kg8 + i] low, K[32b + 8*kg8 + 4 + i] high.
    //
    // Decode: because a column's four K values are four CONSECUTIVE bytes, each 32-bit lane of a
    // decoded plane is exactly one indexed-SDOT operand quad. Two ops per vector suffice:
    //   lsl zA.b, zRaw.b, #4      -> low  nibble, top-justified: 16 * K[k0   .. k0+3]
    //   and zB.d, zRaw.d, z31.d   -> high nibble, in place:      16 * K[k0+4 .. k0+7]
    // (The immediate form of SVE AND is destructive, so the 0xf0 mask is hoisted into z31 to keep
    // this at one instruction.)
    //
    // Both planes are 16x the true weight, so the accumulators start at ZERO and hold exactly
    // 16*dot; `asr #4` undoes that exactly, and only then is the int32 bias added. The mopa sibling
    // cannot use this trick: it seeds bias into ZA via addha before the multiplies, so 16x products
    // would need bias << 4, which overflows the reserved headroom. Here bias is never scaled.
    // Accumulator bound: |16*dot| <= 16 * K * 128 * 8, safe in int32 to K ~ 131000.
    //
    // EIGHT accumulators (z8-z15), not four. This loop is latency-bound on the accumulator chain,
    // never on instruction count or memory: measured at a FLAT 34 cycles per 32-K block from a
    // 128KB working set out to 32MB (IPC 1.50, 2.13 cycles per SDOT). A rate that does not move
    // across four cache levels and into DRAM cannot be bandwidth; it is a dependency chain. With
    // four accumulators the block's 16 SDOTs form 4 chains 4 deep, and 34/4 ~= 8.5 cycles per link
    // is just the indexed-SDOT latency -- the chain WAS the critical path. Eight chains halve the
    // depth to 2, leaving SDOT throughput (16 per block) as the floor. Instruction count is
    // unchanged at 51 per block; only the dependency graph changed. Going past 8 cannot help: at
    // depth 2 the loop is already throughput-bound.
    //
    // SDOT issue order is kg8 = 0, 2, 1, 3 so that the two SDOTs sharing an accumulator are
    // separated by eight others: sets A (z8-z11, kg8 0/1) and B (z12-z15, kg8 2/3) alternate.
    //
    // All ten loads for a block (2 LHS + 8 RHS payload) are issued up front, before any decode, to
    // maximise load-to-use distance and keep ten misses in flight rather than two.
    //
    // Scalar registers: x9 lhs cursor, x10 K left, x11 n done, x12 dst, x13 panel base,
    //   x14 lhs base, x15 payload cursor, x16 K, x17 N, x20 scale offset, x21 block count,
    //   x22 VL bytes, x23 scratch, x24 scale pointer.
    // z0/z1 LHS quad-groups, z16-z23 raw RHS, z2-z7 + z26-z27 decode temps, z8-z15 accumulators,
    //   z24/z25 bias (held across the K loop), z31 nibble mask. z26-z30 are free during the loop
    //   and are reloaded as scale/zero-point/min/max only in the epilogue.
    //
    // d8-d15 MUST be spilled: smstart zeroes the whole of z0-z31, and d8-d15 are the low halves of
    // z8-z15, which base AAPCS64 makes callee-saved. This kernel now genuinely holds its
    // accumulators there, but the spill was mandatory even when it did not -- omitting it silently
    // clobbers the caller's floating-point state on every call.
    stp x20, x21, [sp, -112]!
    stp x22, x23, [sp, 16]
    str x24, [sp, 32]
    stp d8, d9, [sp, 48]
    stp d10, d11, [sp, 64]
    stp d12, d13, [sp, 80]
    stp d14, d15, [sp, 96]
    KAI_ASM_INST(0xd503477f)        // smstart
    ptrue p2.b
    mov z31.b, #-16                 // 0xf0 high-nibble mask (dup imm is signed, -16 == 0xf0)
    ldr x14, [x0, #0x10]            // A_ptr (lhs row)
    ldr x13, [x0, #0x18]            // B_ptr (first packed panel)
    ldr x17, [x0, #0x20]            // N
    ldr x16, [x0, #0x28]            // K
    ldr x12, [x0, #0x30]            // output_ptr
    mov x11, #0x0                   // columns emitted so far
    cntb x22                        // VL in bytes; nr == VL/2 columns
    // x20 = byte offset from a panel's base to its scale region
    //     = nr*4 (bias) + roundup(K,32)*nr/2 (payload) == (2 + roundup(K,32)/4) vectors
    add x23, x16, #31
    lsr x23, x23, #5                // whole 32-K blocks, rounded up
    mov x21, x23                    // block count for the K loop
    lsl x23, x23, #3                // roundup(K,32)/4
    add x23, x23, #2
    mul x20, x23, x22
KAI_ASM_LABEL(label_1)  // N loop: one nr-column panel per iteration
    add x15, x13, x22, LSL #1       // payload starts two vectors (nr*4 bias bytes) in
    ld1w z24.s, p2/z, [x13]                 // bias, n_half 0 -- held for the epilogue
    ld1w z25.s, p2/z, [x13, #1, MUL VL]     // bias, n_half 1
    mov x9, x14                     // lhs cursor
    mov x10, x16                    // K remaining
    mov x23, x21                    // blocks remaining
    // Zero, not bias: the 16x products are shifted back down before the bias is applied.
    // Set A serves kg8 0 and 1, set B serves kg8 2 and 3; within a set the four registers are
    // (n_half 0 low, n_half 1 low, n_half 0 high, n_half 1 high). All eight carry the same 16x
    // top-justification, so they may be summed freely in the epilogue before the single asr #4.
    mov z8.s, #0
    mov z9.s, #0
    mov z10.s, #0
    mov z11.s, #0
    mov z12.s, #0
    mov z13.s, #0
    mov z14.s, #0
    mov z15.s, #0
KAI_ASM_LABEL(label_2)  // K loop: one 32-K block (8 payload vectors) per iteration
    // ---- all ten loads for the block, issued before any dependent work
    // Two LHS quad-groups cover the block: K[32b .. 32b+15] then K[32b+16 .. 32b+31].
    // whilelt bounds each ld1rqb by the K actually left, so no load passes the end of the row;
    // padded RHS nibbles are 0x00 and contribute nothing regardless. The predicate governs only
    // the 16-byte quadword, which is then replicated to every 128-bit segment of the vector.
    whilelt p0.b, xzr, x10
    ld1rqb { z0.b }, p0/z, [x9]
    sub x10, x10, #0x10
    whilelt p3.b, xzr, x10
    ld1rqb { z1.b }, p3/z, [x9, #16]
    sub x10, x10, #0x10
    add x9, x9, #0x20
    ld1b z16.b, p2/z, [x15]                 // kg8 0, n_half 0
    ld1b z17.b, p2/z, [x15, #1, MUL VL]     // kg8 0, n_half 1
    ld1b z18.b, p2/z, [x15, #2, MUL VL]     // kg8 1, n_half 0
    ld1b z19.b, p2/z, [x15, #3, MUL VL]     // kg8 1, n_half 1
    ld1b z20.b, p2/z, [x15, #4, MUL VL]     // kg8 2, n_half 0
    ld1b z21.b, p2/z, [x15, #5, MUL VL]     // kg8 2, n_half 1
    ld1b z22.b, p2/z, [x15, #6, MUL VL]     // kg8 3, n_half 0
    ld1b z23.b, p2/z, [x15, #7, MUL VL]     // kg8 3, n_half 1
    // ---- kg8 = 0 -> set A; LHS z0 quads 0 (K 32b..+3) and 1 (K 32b+4..+7)
    lsl z2.b, z16.b, #4
    and z3.d, z16.d, z31.d
    lsl z4.b, z17.b, #4
    and z5.d, z17.d, z31.d
    sdot z8.s, z2.b, z0.b[0]
    sdot z9.s, z4.b, z0.b[0]
    sdot z10.s, z3.b, z0.b[1]
    sdot z11.s, z5.b, z0.b[1]
    // ---- kg8 = 2 -> set B; LHS z1 quads 0 (K 32b+16..+19) and 1 (K 32b+20..+23)
    lsl z6.b, z20.b, #4
    and z7.d, z20.d, z31.d
    lsl z26.b, z21.b, #4
    and z27.d, z21.d, z31.d
    sdot z12.s, z6.b, z1.b[0]
    sdot z13.s, z26.b, z1.b[0]
    sdot z14.s, z7.b, z1.b[1]
    sdot z15.s, z27.b, z1.b[1]
    // ---- kg8 = 1 -> set A; LHS z0 quads 2 (K 32b+8..+11) and 3 (K 32b+12..+15)
    lsl z2.b, z18.b, #4
    and z3.d, z18.d, z31.d
    lsl z4.b, z19.b, #4
    and z5.d, z19.d, z31.d
    sdot z8.s, z2.b, z0.b[2]
    sdot z9.s, z4.b, z0.b[2]
    sdot z10.s, z3.b, z0.b[3]
    sdot z11.s, z5.b, z0.b[3]
    // ---- kg8 = 3 -> set B; LHS z1 quads 2 (K 32b+24..+27) and 3 (K 32b+28..+31)
    lsl z6.b, z22.b, #4
    and z7.d, z22.d, z31.d
    lsl z26.b, z23.b, #4
    and z27.d, z23.d, z31.d
    sdot z12.s, z6.b, z1.b[2]
    sdot z13.s, z26.b, z1.b[2]
    sdot z14.s, z7.b, z1.b[3]
    sdot z15.s, z27.b, z1.b[3]
    addvl x15, x15, #8
    subs x23, x23, #0x1
    bgt label_2
KAI_ASM_LABEL(label_3)  // Requantise and store
    // Fold the eight chains down to one per n_half. All are 16x scaled, so the sums are too.
    // z26/z27 are dead from here and are reloaded below as the scale vectors.
    add z8.s, z8.s, z10.s
    add z9.s, z9.s, z11.s
    add z12.s, z12.s, z14.s
    add z13.s, z13.s, z15.s
    add z8.s, z8.s, z12.s
    add z9.s, z9.s, z13.s
    // Undo the 16x top-justification (exact: the accumulator is exactly 16*dot), then add bias.
    asr z8.s, z8.s, #4
    asr z9.s, z9.s, #4
    add z8.s, z8.s, z24.s
    add z9.s, z9.s, z25.s
    add x24, x13, x20               // this panel's scale region
    ld1w z26.s, p2/z, [x24]                 // scale, n_half 0
    ld1w z27.s, p2/z, [x24, #1, MUL VL]     // scale, n_half 1
    ld1rw { z28.s }, p2/z, [x0]     // result zero point
    ld1rw { z29.s }, p2/z, [x0, #8] // min
    ld1rw { z30.s }, p2/z, [x0, #4] // max
    scvtf z8.s, p2/m, z8.s
    scvtf z9.s, p2/m, z9.s
    fmul z8.s, z8.s, z26.s
    fmul z9.s, z9.s, z27.s
    frintn z8.s, p2/m, z8.s
    frintn z9.s, p2/m, z9.s
    fcvtzs z8.s, p2/m, z8.s
    fcvtzs z9.s, p2/m, z9.s
    add z8.s, z8.s, z28.s
    add z9.s, z9.s, z28.s
    sclamp z8.s, z29.s, z30.s
    sclamp z9.s, z29.s, z30.s
    // Interleave the halves back into column order; st1b of an .h vector emits nr bytes.
    whilelt p1.h, x11, x17
    uzp1 z8.h, z8.h, z9.h
    st1b { z8.h }, p1, [x12]
    add x13, x13, x20               // next panel: past the payload ...
    addvl x13, x13, #2              // ... and past this one's nr*4 scale bytes
    add x12, x12, x22, LSR #1       // dst advances nr == VL/2 bytes
    incw x11, ALL, MUL #2           // nr columns emitted (2 * VL/4)
    cmp x11, x17
    blt label_1
    KAI_ASM_INST(0xd503467f)        // smstop
    ldp x22, x23, [sp, 16]
    ldr x24, [sp, 32]
    ldp d8, d9, [sp, 48]
    ldp d10, d11, [sp, 64]
    ldp d12, d13, [sp, 80]
    ldp d14, d15, [sp, 96]
    ldp x20, x21, [sp], 112
    ret
    KAI_ASM_FUNCTION_END(kai_kernel_matmul_clamp_qai8_qai8_qsi4cxp8vsx4s4s0sf32bi32_1x8vs_qmx_dot)

    KAI_ASM_END
