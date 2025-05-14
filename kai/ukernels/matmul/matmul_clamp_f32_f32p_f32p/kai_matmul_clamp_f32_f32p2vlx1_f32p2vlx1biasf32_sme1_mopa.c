//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Do not flag up inline assembly blocks
#pragma GCC diagnostic ignored "-Woverlength-strings"

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE2)
#error This file must be compiled for AArch64, FEAT_SVE2.
#else  // Architectural features check.

#include "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa.h"
__asm__(".include \"helper_macros_matmul_clamp_f32_f32p_f32p.S\"");

#include <stddef.h>
#include <stdint.h>

#include "kai/kai_common.h"

static const size_t kai_mr = 2;
static const size_t kai_nr = 2;
static const size_t kai_kr = 1;
static const size_t kai_sr = 1;

size_t kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa(void) {
    return kai_mr * kai_get_sme_vector_length_u32();
}

size_t kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa(void) {
    return kai_nr * kai_get_sme_vector_length_u32();
}

size_t kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa(void) {
    return kai_kr;
}

size_t kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa(void) {
    return kai_sr;
}

size_t kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa(size_t m_idx, size_t k) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa() == 0);
    return m_idx * k * sizeof(float);
}

size_t kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa(size_t n_idx, size_t k) {
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa() == 0);
    return n_idx * (k * sizeof(float) + sizeof(float));
}

size_t kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa(
    size_t m_idx, size_t n_idx, size_t dst_stride) {
    KAI_ASSUME(m_idx % kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa() == 0);
    KAI_ASSUME(n_idx % kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa() == 0);

    return m_idx * dst_stride + n_idx * sizeof(float);
}

size_t kai_get_dst_size_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa(size_t m, size_t n) {
    return m * n * sizeof(float);
}

void kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa(
    size_t m, size_t n, size_t k, const void* lhs_packed, const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, float clamp_min, float clamp_max) {
    KAI_ASSUME(dst_stride_col == sizeof(float));

    typedef struct {
        const void* A;
        const void* B;

        void* C;
        uint64_t ldcb;
        uint64_t M, N, K;
        float min;
        float max;

        void* accumulator_buffer;
        uint64_t flags;
    } KernelArgs;

    KernelArgs args;

    args.A = lhs_packed;
    args.B = rhs_packed;

    args.C = dst;
    args.ldcb = dst_stride_row;
    args.M = m;
    args.N = n;
    args.K = k;
    args.min = clamp_min;
    args.max = clamp_max;

    args.accumulator_buffer = NULL;
    args.flags = 0;

    __asm__ __volatile__(
        "ldr x17, [%x[args], %[offsetof_flags]]\n"
        "SMSTART\n"
        "ptrue p0.b\n"
        "ptrue p7.b\n"
        "ldr x16, [%x[args], %[offsetof_accumulator_buffer]]\n"
        "ldr x15, [%x[args], %[offsetof_accumulator_buffer]]\n"
        "tbz x17, #0, 2f\n"
        "mov x12, #0x0\n"
        "cntw x20\n"
        "1:"  // Initial accumulator load from buffer: Loop
        "load_data z24.s, z25.s, z26.s, z27.s, p7, x16\n"
        "load_data z12.s, z13.s, z14.s, z15.s, p7, x16\n"
        "load_data z0.s, z1.s, z2.s, z3.s, p7, x16\n"
        "load_data z16.s, z17.s, z18.s, z19.s, p7, x16\n"
        "move_vector_tile za0h.s, z24.s, z25.s, z26.s, z27.s, p7, w12\n"
        "move_vector_tile za1h.s, z12.s, z13.s, z14.s, z15.s, p7, w12\n"
        "move_vector_tile za2h.s, z0.s, z1.s, z2.s, z3.s, p7, w12\n"
        "move_vector_tile za3h.s, z16.s, z17.s, z18.s, z19.s, p7, w12\n"
        "add x12, x12, #0x4\n"
        "cmp x12, x20\n"
        "blt 1b\n"
        "2:"  // Initial accumulator load from buffer: End
        "ldr w14, [%x[args], %[offsetof_M]]\n"
        "mov x13, #0x0\n"
        "mov x11, #0x0\n"
        "ldr w10, [%x[args], %[offsetof_N]]\n"
        "ldr x9, [%x[args], %[offsetof_A]]\n"
        "3:"  // M loop
        "ldr x28, [%x[args], %[offsetof_B]]\n"
        "4:"  // N loop
        "mov x27, x9\n"
        "whilelt p4.s, x11, x10\n"
        "tbnz x17, #0, 5f\n"
        "fmov z17.s, #1.0\n"
        "load_data_pred_index_counter_2 z10.s, z11.s, p4, p4.s, x11, x10, x28, x8 \n"  // Load bias
        "zero { za }\n"
        "addvl x28, x28, #2\n"
        "fmopa za0.s, p0/M, p0/M, z17.s, z10.s\n"
        "fmopa za1.s, p0/M, p0/M, z17.s, z11.s\n"
        "fmopa za2.s, p0/M, p0/M, z17.s, z10.s\n"
        "fmopa za3.s, p0/M, p0/M, z17.s, z11.s\n"
        "5:"  // Prepare accumulators: Test for last block
        "mov x20, x11\n"
        "mov x21, x13\n"
        "incw x20, ALL, MUL #2\n"
        "incw x21, ALL, MUL #2\n"
        "cmp x20, x10\n"
        "mov x20, x17\n"
        "csel x21, x13, x21, LT\n"
        "bfm x17, XZR, #0x0, #0x0  // bfc x17, #0x0, #0x1\n"
        "cmp x21, x14\n"
        "csel x17, x20, x17, LT\n"
        "ldr x20, [%x[args], %[offsetof_K]]\n"
        "lsr x21, x20, #0x2\n"
        "and x20, x20, #0x3\n"
        "cbz x21, 9f\n"
        "subs x21, x21, #0x1\n"
        "load_data_2 z22.s, z23.s, p7, x27\n"
        "load_data_2 z7.s, z15.s, p7, x28\n"
        "load_data_2 z6.s, z14.s, p7, x27\n"
        "load_data_2 z20.s, z21.s, p7, x28\n"
        "load_data_2 z2.s, z10.s, p7, x27\n"
        "load_data_2 z3.s, z11.s, p7, x28\n"
        "load_data_2 z1.s, z9.s, p7, x27\n"
        "load_data_2 z4.s, z5.s, p7, x28\n"
        "ble 8f\n"
        "7:"  // K loop
        "fmopa za0.s, p0/M, p0/M, z22.s, z7.s\n"
        "subs x21, x21, #0x1\n"
        "fmopa za1.s, p0/M, p0/M, z22.s, z15.s\n"
        "fmopa za2.s, p0/M, p0/M, z23.s, z7.s\n"
        "fmopa za3.s, p0/M, p0/M, z23.s, z15.s\n"
        "load_data_2 z22.s, z23.s, p7, x27\n"
        "fmopa za0.s, p0/M, p0/M, z6.s, z20.s\n"
        "load_data_2 z7.s, z15.s, p7, x28\n"
        "fmopa za1.s, p0/M, p0/M, z6.s, z21.s\n"
        "fmopa za2.s, p0/M, p0/M, z14.s, z20.s\n"
        "fmopa za3.s, p0/M, p0/M, z14.s, z21.s\n"
        "load_data_2 z6.s, z14.s, p7, x27\n"
        "fmopa za0.s, p0/M, p0/M, z2.s, z3.s\n"
        "load_data_2 z20.s, z21.s, p7, x28\n"
        "fmopa za1.s, p0/M, p0/M, z2.s, z11.s\n"
        "fmopa za2.s, p0/M, p0/M, z10.s, z3.s\n"
        "fmopa za3.s, p0/M, p0/M, z10.s, z11.s\n"
        "load_data_2 z2.s, z10.s, p7, x27\n"
        "load_data_2 z3.s, z11.s, p7, x28\n"
        "fmopa za0.s, p0/M, p0/M, z1.s, z4.s\n"
        "fmopa za1.s, p0/M, p0/M, z1.s, z5.s\n"
        "fmopa za2.s, p0/M, p0/M, z9.s, z4.s\n"
        "fmopa za3.s, p0/M, p0/M, z9.s, z5.s\n"
        "load_data_2 z1.s, z9.s, p7, x27\n"
        "load_data_2 z4.s, z5.s, p7, x28\n"
        "bgt 7b\n"
        "8:"  // K loop tail
        "fmopa za0.s, p0/M, p0/M, z22.s, z7.s\n"
        "fmopa za1.s, p0/M, p0/M, z22.s, z15.s\n"
        "fmopa za2.s, p0/M, p0/M, z23.s, z7.s\n"
        "fmopa za3.s, p0/M, p0/M, z23.s, z15.s\n"
        "fmopa za0.s, p0/M, p0/M, z6.s, z20.s\n"
        "fmopa za1.s, p0/M, p0/M, z6.s, z21.s\n"
        "fmopa za2.s, p0/M, p0/M, z14.s, z20.s\n"
        "fmopa za3.s, p0/M, p0/M, z14.s, z21.s\n"
        "fmopa za0.s, p0/M, p0/M, z2.s, z3.s\n"
        "fmopa za1.s, p0/M, p0/M, z2.s, z11.s\n"
        "fmopa za2.s, p0/M, p0/M, z10.s, z3.s\n"
        "fmopa za3.s, p0/M, p0/M, z10.s, z11.s\n"
        "fmopa za0.s, p0/M, p0/M, z1.s, z4.s\n"
        "fmopa za1.s, p0/M, p0/M, z1.s, z5.s\n"
        "fmopa za2.s, p0/M, p0/M, z9.s, z4.s\n"
        "fmopa za3.s, p0/M, p0/M, z9.s, z5.s\n"
        "9:"  // K oddments
        "cbz x20, 11f\n"
        "10:"  // K oddments: Loop
        "load_data_2 z10.s, z11.s, p7, x27\n"
        "subs x20, x20, #0x1\n"
        "load_data_2 z14.s, z15.s, p7, x28\n"
        "fmopa za0.s, p0/M, p0/M, z10.s, z14.s\n"
        "fmopa za1.s, p0/M, p0/M, z10.s, z15.s\n"
        "fmopa za2.s, p0/M, p0/M, z11.s, z14.s\n"
        "fmopa za3.s, p0/M, p0/M, z11.s, z15.s\n"
        "bgt 10b\n"
        "11:"  // K oddments: End
        "tbz x17, #1, 15f\n"
        "tbz x17, #0, 13f\n"
        "mov x12, #0x0\n"
        "cntw x20\n"
        "12:"  // Store to partial result buffer: Store and refill: Loop
        "load_data z0.s, z1.s, z2.s, z3.s, p7, x16\n"
        "move_tile_vector za0h.s, z20.s, z21.s, z22.s, z23.s, p7, w12\n"
        "move_tile_vector za1h.s, z28.s, z29.s, z30.s, z31.s, p7, w12\n"
        "load_data z4.s, z5.s, z6.s, z7.s, p7, x16\n"
        "move_tile_vector za2h.s, z8.s, z9.s, z10.s, z11.s, p7, w12\n"
        "move_tile_vector za3h.s, z12.s, z13.s, z14.s, z15.s, p7, w12\n"
        "load_data z16.s, z17.s, z18.s, z19.s, p7, x16\n"
        "load_data z24.s, z25.s, z26.s, z27.s, p7, x16\n"
        "move_vector_tile za0h.s, z0.s, z1.s, z2.s, z3.s, p7, w12\n"
        "move_vector_tile za1h.s, z4.s, z5.s, z6.s, z7.s, p7, w12\n"
        "store_data z20.s, z21.s, z22.s, z23.s, p7, x15\n"
        "move_vector_tile za2h.s, z16.s, z17.s, z18.s, z19.s, p7, w12\n"
        "store_data z28.s, z29.s, z30.s, z31.s, p7, x15\n"
        "move_vector_tile za3h.s, z24.s, z25.s, z26.s, z27.s, p7, w12\n"
        "add x12, x12, #0x4\n"
        "store_data z8.s, z9.s, z10.s, z11.s, p7, x15\n"
        "cmp x12, x20\n"
        "store_data z12.s, z13.s, z14.s, z15.s, p7, x15\n"
        "blt 12b\n"
        "b 31f\n"
        "13:"  // Store to partial result buffer: Store only
        "mov x12, #0x0\n"
        "cntw x20\n"
        "14:"  // Store to partial result buffer: Store only: Loop
        "move_tile_vector za0h.s, z0.s, z1.s, z2.s, z3.s, p7, w12\n"
        "move_tile_vector za1h.s, z16.s, z17.s, z18.s, z19.s, p7, w12\n"
        "move_tile_vector za2h.s, z28.s, z29.s, z30.s, z31.s, p7, w12\n"
        "move_tile_vector za3h.s, z20.s, z21.s, z22.s, z23.s, p7, w12\n"
        "store_data z0.s, z1.s, z2.s, z3.s, p7, x15\n"
        "add x12, x12, #0x4\n"
        "store_data z16.s, z17.s, z18.s, z19.s, p7, x15\n"
        "cmp x12, x20\n"
        "store_data z28.s, z29.s, z30.s, z31.s, p7, x15\n"
        "store_data z20.s, z21.s, z22.s, z23.s, p7, x15\n"
        "blt 14b\n"
        "b 31f\n"
        "15:"  // Store to output array
        "ldr x26, [%x[args], %[offsetof_C]]\n"
        "sub x25, x14, x13\n"
        "ldr x24, [%x[args], %[offsetof_ldcb]]\n"
        "add x26, x26, x11, LSL #2\n"  // C += n
        "madd x26, x13, x24, x26\n"    // C += m * ldc
        "tbz x17, #2, 22f\n"
        "cntw x23\n"
        "mov x12, #0x0\n"
        "cmp x25, x23\n"
        "csel x22, x25, x23, LT\n"
        "lsr x21, x22, #0x2\n"
        "and x20, x22, #0x3\n"
        "cbz x21, 17f\n"
        "16:"  // Store to output array: Skip activation: Accumulator row 0 loop
        "move_tile_vector za0h.s, z4.s, z5.s, z6.s, z7.s, p7, w12\n"
        "move_tile_vector za1h.s, z12.s, z13.s, z14.s, z15.s, p7, w12\n"
        "store_data_pred_index_counter_2 z4.s, z12.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "add x12, x12, #0x4\n"
        "store_data_pred_index_counter_2 z5.s, z13.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "store_data_pred_index_counter_2 z6.s, z14.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "store_data_pred_index_counter_2 z7.s, z15.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "cmp x12, x21, LSL #2\n"
        "blt 16b\n"
        "17:"  // Store to output array: Skip activation: Accumulator row 0 oddments
        "cbz x20, 18f\n"
        "move_tile_vector za0h.s, z0.s, z1.s, z2.s, z3.s, p7, w12\n"
        "move_tile_vector za1h.s, z8.s, z9.s, z10.s, z11.s, p7, w12\n"
        "store_data_pred_index_counter_2 z0.s, z8.s, p4, p4.s, x11, x10, x26, x8 \n"
        "subs x20, x20, #0x1\n"
        "add x26, x26, x24\n"
        "beq 18f\n"
        "store_data_pred_index_counter_2 z1.s, z9.s, p4, p4.s, x11, x10, x26, x8 \n"
        "subs x20, x20, #0x1\n"
        "add x26, x26, x24\n"
        "beq 18f\n"
        "store_data_pred_index_counter_2 z2.s, z10.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "18:"  // Store to output array: Skip activation: Accumulator row 0 oddments: End
        "subs x25, x25, x22\n"
        "beq 22f\n"
        "cmp x25, x23\n"
        "mov x12, #0x0\n"
        "csel x22, x25, x23, LT\n"
        "lsr x21, x22, #0x2\n"
        "and x20, x22, #0x3\n"
        "cbz x21, 20f\n"
        "19:"  // Store to output array: Skip activation: Accumulator row 1 loop
        "move_tile_vector za2h.s, z4.s, z5.s, z6.s, z7.s, p7, w12\n"
        "move_tile_vector za3h.s, z12.s, z13.s, z14.s, z15.s, p7, w12\n"
        "store_data_pred_index_counter_2 z4.s, z12.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "add x12, x12, #0x4\n"
        "store_data_pred_index_counter_2 z5.s, z13.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "store_data_pred_index_counter_2 z6.s, z14.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "store_data_pred_index_counter_2 z7.s, z15.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "cmp x12, x21, LSL #2\n"
        "blt 19b\n"
        "20:"  // Store to output array: Skip activation: Accumulator row 1 oddments
        "cbz x20, 21f\n"
        "move_tile_vector za2h.s, z4.s, z5.s, z6.s, z7.s, p7, w12\n"
        "move_tile_vector za3h.s, z12.s, z13.s, z14.s, z15.s, p7, w12\n"
        "store_data_pred_index_counter_2 z4.s, z12.s, p4, p4.s, x11, x10, x26, x8 \n"
        "subs x20, x20, #0x1\n"
        "add x26, x26, x24\n"
        "beq 21f\n"
        "store_data_pred_index_counter_2 z5.s, z13.s, p4, p4.s, x11, x10, x26, x8 \n"
        "subs x20, x20, #0x1\n"
        "add x26, x26, x24\n"
        "beq 21f\n"
        "store_data_pred_index_counter_2 z6.s, z14.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "21:"  // Store to output array: Skip activation: Accumulator row 1 oddments: End
        "subs x25, x25, x22\n"
        "beq 22f\n"
        "b 29f\n"
        "22:"  // Store to output array: Skip activation: End
        "cntw x23\n"
        "ld1rw { z21.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_min]]\n"
        "mov x12, #0x0\n"
        "cmp x25, x23\n"
        "ld1rw { z20.s }, p0/Z, [%x[args], %[offsetof_KernelArgs_max]]\n"
        "csel x22, x25, x23, LT\n"
        "lsr x21, x22, #0x2\n"
        "and x20, x22, #0x3\n"
        "cbz x21, 24f\n"
        "23:"  // Store to output array: Accumulator row 0 loop
        "move_tile_vector za0h.s, z16.s, z17.s, z18.s, z19.s, p7, w12\n"
        "move_tile_vector za1h.s, z24.s, z25.s, z26.s, z27.s, p7, w12\n"
        "clamp_float z16.s, z20.s, z21.s, p7\n"
        "clamp_float z17.s, z20.s, z21.s, p7\n"
        "clamp_float z18.s, z20.s, z21.s, p7\n"
        "clamp_float z19.s, z20.s, z21.s, p7\n"
        "clamp_float z24.s, z20.s, z21.s, p7\n"
        "clamp_float z25.s, z20.s, z21.s, p7\n"
        "clamp_float z26.s, z20.s, z21.s, p7\n"
        "clamp_float z27.s, z20.s, z21.s, p7\n"
        "add x12, x12, #0x4\n"
        "store_data_pred_index_counter_2 z16.s, z24.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "store_data_pred_index_counter_2 z17.s, z25.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "store_data_pred_index_counter_2 z18.s, z26.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "store_data_pred_index_counter_2 z19.s, z27.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "cmp x12, x21, LSL #2\n"
        "blt 23b\n"
        "24:"  // Store to output array: Accumulator row 0 oddments
        "cbz x20, 25f\n"
        "move_tile_vector za0h.s, z16.s, z17.s, z18.s, z19.s, p7, w12\n"
        "move_tile_vector za1h.s, z24.s, z25.s, z26.s, z27.s, p7, w12\n"
        "clamp_float z16.s, z20.s, z21.s, p7\n"
        "clamp_float z17.s, z20.s, z21.s, p7\n"
        "clamp_float z18.s, z20.s, z21.s, p7\n"
        "clamp_float z19.s, z20.s, z21.s, p7\n"
        "clamp_float z24.s, z20.s, z21.s, p7\n"
        "clamp_float z25.s, z20.s, z21.s, p7\n"
        "clamp_float z26.s, z20.s, z21.s, p7\n"
        "clamp_float z27.s, z20.s, z21.s, p7\n"
        "store_data_pred_index_counter_2 z16.s, z24.s, p4, p4.s, x11, x10, x26, x8 \n"
        "subs x20, x20, #0x1\n"
        "add x26, x26, x24\n"
        "beq 25f\n"
        "store_data_pred_index_counter_2 z17.s, z25.s, p4, p4.s, x11, x10, x26, x8 \n"
        "subs x20, x20, #0x1\n"
        "add x26, x26, x24\n"
        "beq 25f\n"
        "store_data_pred_index_counter_2 z18.s, z26.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "25:"  // Store to output array: Accumulator row 0 oddments: End
        "subs x25, x25, x22\n"
        "beq 29f\n"
        "cmp x25, x23\n"
        "mov x12, #0x0\n"
        "csel x20, x25, x23, LT\n"
        "lsr x21, x20, #0x2\n"
        "and x20, x20, #0x3\n"
        "cbz x21, 27f\n"
        "26:"  // Store to output array: Accumulator row 1 loop
        "move_tile_vector za2h.s, z0.s, z1.s, z2.s, z3.s, p7, w12\n"
        "move_tile_vector za3h.s, z8.s, z9.s, z10.s, z11.s, p7, w12\n"
        "clamp_float z0.s, z20.s, z21.s, p7\n"
        "clamp_float z1.s, z20.s, z21.s, p7\n"
        "clamp_float z2.s, z20.s, z21.s, p7\n"
        "clamp_float z3.s, z20.s, z21.s, p7\n"
        "clamp_float z8.s, z20.s, z21.s, p7\n"
        "clamp_float z9.s, z20.s, z21.s, p7\n"
        "clamp_float z10.s, z20.s, z21.s, p7\n"
        "clamp_float z11.s, z20.s, z21.s, p7\n"
        "add x12, x12, #0x4\n"
        "store_data_pred_index_counter_2 z0.s, z8.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "store_data_pred_index_counter_2 z1.s, z9.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "store_data_pred_index_counter_2 z2.s, z10.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "store_data_pred_index_counter_2 z3.s, z11.s, p4, p4.s, x11, x10, x26, x8 \n"
        "add x26, x26, x24\n"
        "cmp x12, x21, LSL #2\n"
        "blt 26b\n"
        "27:"  // Store to output array: Accumulator row 1 oddments
        "cbz x20, 28f\n"
        "move_tile_vector za2h.s, z16.s, z17.s, z18.s, z19.s, p7, w12\n"
        "move_tile_vector za3h.s, z24.s, z25.s, z26.s, z27.s, p7, w12\n"
        "clamp_float z16.s, z20.s, z21.s, p7\n"
        "clamp_float z17.s, z20.s, z21.s, p7\n"
        "clamp_float z18.s, z20.s, z21.s, p7\n"
        "clamp_float z19.s, z20.s, z21.s, p7\n"
        "clamp_float z24.s, z20.s, z21.s, p7\n"
        "clamp_float z25.s, z20.s, z21.s, p7\n"
        "clamp_float z26.s, z20.s, z21.s, p7\n"
        "clamp_float z27.s, z20.s, z21.s, p7\n"
        "store_data_pred_index_counter_2 z16.s, z24.s, p4, p4.s, x11, x10, x26, x8 \n"
        "subs x20, x20, #0x1\n"
        "add x26, x26, x24\n"
        "beq 28f\n"
        "store_data_pred_index_counter_2 z17.s, z25.s, p4, p4.s, x11, x10, x26, x8 \n"
        "subs x20, x20, #0x1\n"
        "add x26, x26, x24\n"
        "beq 28f\n"
        "store_data_pred_index_counter_2 z18.s, z26.s, p4, p4.s, x11, x10, x26, x8 \n"
        "28:"  // Store to output array: Accumulator row 1 oddments: End
        "29:"  // Store to output array: End
        "tbz x17, #0, 31f\n"
        "mov x12, #0x0\n"
        "cntw x20\n"
        "30:"  // Store to output array: Refill accumulators: Loop
        "load_data z8.s, z9.s, z10.s, z11.s, p7, x16\n"
        "load_data z0.s, z1.s, z2.s, z3.s, p7, x16\n"
        "load_data z4.s, z5.s, z6.s, z7.s, p7, x16\n"
        "load_data z12.s, z13.s, z14.s, z15.s, p7, x16\n"
        "move_vector_tile za0h.s, z8.s, z9.s, z10.s, z11.s, p7, w12\n"
        "move_vector_tile za1h.s, z0.s, z1.s, z2.s, z3.s, p7, w12\n"
        "move_vector_tile za2h.s, z4.s, z5.s, z6.s, z7.s, p7, w12\n"
        "move_vector_tile za3h.s, z12.s, z13.s, z14.s, z15.s, p7, w12\n"
        "add x12, x12, #0x4\n"
        "cmp x12, x20\n"
        "blt 30b\n"
        "31:"  // End block
        "incw x11, ALL, MUL #2\n"
        "cmp x11, x10\n"
        "blt 4b\n"
        "incw x13, ALL, MUL #2\n"
        "mov x11, #0x0\n"
        "cmp x13, x14\n"
        "mov x9, x27\n"
        "blt 3b\n"
        "SMSTOP\n"
        :
        : [args] "r"(&args), [offsetof_A] "I"(offsetof(KernelArgs, A)), [offsetof_B] "I"(offsetof(KernelArgs, B)),
          [offsetof_C] "I"(offsetof(KernelArgs, C)), [offsetof_K] "I"(offsetof(KernelArgs, K)),
          [offsetof_KernelArgs_max] "I"(offsetof(KernelArgs, max)),
          [offsetof_KernelArgs_min] "I"(offsetof(KernelArgs, min)), [offsetof_M] "I"(offsetof(KernelArgs, M)),
          [offsetof_N] "I"(offsetof(KernelArgs, N)),
          [offsetof_accumulator_buffer] "I"(offsetof(KernelArgs, accumulator_buffer)),
          [offsetof_flags] "I"(offsetof(KernelArgs, flags)), [offsetof_ldcb] "I"(offsetof(KernelArgs, ldcb))
        : "cc", "memory", "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "p11", "p12", "p13", "p14",
          "p15","x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x20", "x21", "x22", "x23", "x24", "x25",
          "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13",
          "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28",
          "z29", "z30", "z31");
}

#endif  // Architectural features check.
