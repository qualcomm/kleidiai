//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
#if !defined(__ARM_FEATURE_DOTPROD) && !defined(__ARM_FEATURE_MATMUL_INT8)
#error "Dotprod and I8mm extensions required to compile this example"
#else
#include <cassert>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

// Include micro-kernel variants
#include "kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm.h"
#include "kai_matmul_clamp_f32_qai8dxp_qsi4c32p_interface.h"
#include "kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0.h"
#include "kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0.h"

// QMX MOPA kernel
#include "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa.h"
#include "kai_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon.h"
#include "kai_rhs_pack_kxn_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon.h"

// ─── Thread count parsing ────────────────────────────────────────────────────

static size_t parse_thread_count_value(const std::string& value) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0') {
        std::cerr << "Invalid thread count: '" << value << "'\n";
        std::exit(EXIT_FAILURE);
    }
    if (parsed == 0 || parsed > std::numeric_limits<size_t>::max()) {
        std::cerr << "Thread count must be in range [1, " << std::numeric_limits<size_t>::max() << "]\n";
        std::exit(EXIT_FAILURE);
    }
    return static_cast<size_t>(parsed);
}

static void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [--threads <count> | --threads=<count>]\n";
}

static size_t parse_thread_count(int argc, char** argv) {
    size_t thread_count = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (arg == "--threads" || arg == "-t") {
            if ((i + 1) >= argc) {
                std::cerr << "--threads expects a value\n";
                print_usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            thread_count = parse_thread_count_value(argv[++i]);
            continue;
        }
        const std::string prefix = "--threads=";
        if (arg.rfind(prefix, 0) == 0) {
            thread_count = parse_thread_count_value(arg.substr(prefix.size()));
            continue;
        }
        std::cerr << "Unrecognized argument: " << arg << "\n";
        print_usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }
    return thread_count;
}

#define INT4_MIN (-8)
#define INT4_MAX (7)

enum class rhs_format {
    nxk,
    kxn,
};
struct mnk {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    size_t bl = 0;
};
mnk matmul_shapes[] = {{1, 33, 32, 32}, {13, 33, 32, 32}, {37, 75, 256, 64}, {16, 32, 64, 32}, {8, 32, 64, 64}};
// Micro-kernel interface
struct kai_matmul_ukernel_f32_qa8dxp_qs4c32p {
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel ukernel;
    std::string name = {};
};

kai_matmul_ukernel_f32_qa8dxp_qs4c32p ukernel_variants[] = {
    {{kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod,
      kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod},
     "matmul_clamp_f32_qai8dxp1x8_qsi4c32p4x8_1x4x32_neon_dotprod"},
    {{kai_get_m_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
      kai_get_n_step_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
      kai_get_mr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
      kai_get_nr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
      kai_get_kr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
      kai_get_sr_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
      kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
      kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
      kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
      kai_get_dst_size_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod,
      kai_run_matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod},
     "matmul_clamp_f32_qai8dxp1x8_qsi4c32p8x8_1x8x32_neon_dotprod"},
    {{kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
      kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
      kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
      kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
      kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
      kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
      kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
      kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
      kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
      kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm,
      kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm},
     "matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_8x4x32_neon_i8mm"},
    {{kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm,
      kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm},
     "matmul_clamp_f32_qai8dxp4x8_qsi4c32p4x8_16x4x32_neon_i8mm"},
    {{kai_get_m_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
      kai_get_n_step_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
      kai_get_mr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
      kai_get_nr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
      kai_get_kr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
      kai_get_sr_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
      kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
      kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
      kai_get_dst_offset_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
      kai_get_dst_size_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm,
      kai_run_matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm},
     "matmul_clamp_f32_qai8dxp4x8_qsi4c32p8x8_4x8x32_neon_i8mm"},
    {{kai_get_m_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_n_step_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_kr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_sr_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_dst_offset_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_get_dst_size_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod,
      kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod},
     "matmul_clamp_f32_qai8dxp1x4_qsi4c32p4x4_1x4_neon_dotprod"},
};

// Number of micro-kernel variants stored in the array
const size_t num_ukernel_variants = sizeof(ukernel_variants) / sizeof(ukernel_variants[0]);

// ─── QMX MOPA ukernel variant table ─────────────────────────────────────────

struct kai_matmul_ukernel_f32_qa8dxp_qs4c32p_qmx {
    kai_matmul_clamp_f32_qai8dxp_qsi4c32p_ukernel ukernel;
    std::string name = {};
};

static kai_matmul_ukernel_f32_qa8dxp_qs4c32p_qmx qmx_ukernel_variants[] = {
    {{kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa,
      kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa,
      kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa,
      kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa,
      kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa,
      kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa,
      kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa,
      kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa,
      kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa,
      kai_get_dst_size_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa,
      kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa},
     "matmul_clamp_f32_qai8dxp1vlx4_qsi4c32p4vlx4_1vlx4vl_qmx_mopa"},
};

static const size_t num_qmx_ukernel_variants =
    sizeof(qmx_ukernel_variants) / sizeof(qmx_ukernel_variants[0]);

static size_t roundup(size_t a, size_t b) {
    return ((a + b - 1) / b) * b;
}

static inline size_t get_num_blocks_per_row(size_t k, size_t bl) {
    return roundup(k, bl) / bl;
}

static inline size_t get_rhs_native_stride(size_t x) {
    return roundup(x, 2) / 2;
}

static inline size_t get_rhs_scale_stride(size_t k, size_t bl) {
    const size_t num_blocks_per_row = get_num_blocks_per_row(k, bl);
    return num_blocks_per_row * sizeof(uint16_t);
}

static void fill_uniform_random(size_t num_rows, size_t num_cols, float* dst, size_t seed) {
    std::srand(seed);

    // Fill the array with random values between -1 and 1
    for (size_t i = 0; i < num_rows * num_cols; i++) {
        dst[i] = (float)((double)std::rand() / RAND_MAX) * 2 - 1;
    }
}

static void quant_nxk_qs4c32_f32(
    size_t n, size_t k, size_t bl, const float* rhs_f32, uint8_t* rhs_qs4c32, uint16_t* rhs_scales_bf16) {
    const size_t num_blocks_row = get_num_blocks_per_row(k, bl);
    const size_t rhs_qs4c32_stride = get_rhs_native_stride(k);

    // Make sure the output is filled with zeros
    std::memset(rhs_qs4c32, 0, n * rhs_qs4c32_stride);

    for (size_t row_idx = 0; row_idx < n; ++row_idx) {
        const float* src_ptr = rhs_f32 + row_idx * k;

        for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
            float amax = 0.0f;
            float max = 0.0f;

            for (size_t b = 0; b < bl; ++b) {
                const size_t k_idx = block_idx * bl + b;

                if (k_idx >= k) {
                    break;
                }

                const float src0_0 = src_ptr[k_idx];
                const float asrc0_0 = fabsf(src0_0);

                if (amax < asrc0_0) {
                    amax = asrc0_0;
                    max = src0_0;
                }
            }

            const float scale = max / -8.0;
            const float recip_scale = scale ? 1.0f / scale : 0.0f;

            // Store the scale in the dedicated buffer
            *rhs_scales_bf16 = kai_cast_bf16_f32(scale);

            rhs_scales_bf16 += 1;

            for (size_t i = 0; i < bl; ++i) {
                const size_t k_idx = block_idx * bl + i;

                if (k_idx >= k) {
                    break;
                }

                const float src0_0 = src_ptr[k_idx];

                // Scale the values
                int32_t v0_s32 = (int32_t)(round(src0_0 * recip_scale));

                // Maximum/minimum int4 values
                v0_s32 = std::max(v0_s32, INT4_MIN);
                v0_s32 = std::min(v0_s32, INT4_MAX);

                const uint8_t v0_u8 = (uint8_t)(v0_s32 + 8);

                const size_t dst_addr = (k_idx / 2) + row_idx * rhs_qs4c32_stride;
                uint8_t rhs_v0 = rhs_qs4c32[dst_addr];

                if ((k_idx % 2) == 0) {
                    rhs_v0 = v0_u8;
                } else {
                    rhs_v0 |= (v0_u8 << 4);
                }

                rhs_qs4c32[dst_addr] = rhs_v0;
            }
        }
    }
}

static void quant_kxn_qs4c32_f32(
    size_t n, size_t k, size_t bl, const float* rhs_f32, uint8_t* rhs_qs4c32, uint16_t* rhs_scales_bf16) {
    const size_t num_blocks_row = get_num_blocks_per_row(k, bl);
    const size_t rhs_qs4c32_stride = get_rhs_native_stride(n);

    // Make sure the output is filled with zeros
    std::memset(rhs_qs4c32, 0, k * rhs_qs4c32_stride);

    for (size_t row_idx = 0; row_idx < n; ++row_idx) {
        const float* src_ptr = rhs_f32 + row_idx * k;

        for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
            float amax = 0.0f;
            float max = 0.0f;

            for (size_t b = 0; b < bl; ++b) {
                const size_t k_idx = block_idx * bl + b;

                if (k_idx >= k) {
                    break;
                }

                const float src0_0 = src_ptr[k_idx];
                const float asrc0_0 = fabsf(src0_0);

                if (amax < asrc0_0) {
                    amax = asrc0_0;
                    max = src0_0;
                }
            }

            const float scale = max / -8.0;
            const float recip_scale = scale ? 1.0f / scale : 0.0f;

            // Store the scale in the dedicated buffer
            *rhs_scales_bf16 = kai_cast_bf16_f32(scale);

            rhs_scales_bf16 += 1;

            for (size_t i = 0; i < bl; ++i) {
                const size_t k_idx = block_idx * bl + i;

                if (k_idx >= k) {
                    break;
                }

                const float src0_0 = src_ptr[k_idx];

                // Scale the values
                int32_t v0_s32 = (int32_t)(round(src0_0 * recip_scale));

                // Maximum/minimum int4 values
                v0_s32 = std::max(v0_s32, INT4_MIN);
                v0_s32 = std::min(v0_s32, INT4_MAX);

                const uint8_t v0_u8 = (uint8_t)(v0_s32 + 8);

                const size_t dst_addr = (row_idx / 2) + k_idx * rhs_qs4c32_stride;
                uint8_t rhs_v0 = rhs_qs4c32[dst_addr];

                if ((row_idx % 2) == 0) {
                    rhs_v0 = v0_u8;
                } else {
                    rhs_v0 |= (v0_u8 << 4);
                }

                rhs_qs4c32[dst_addr] = rhs_v0;
            }
        }
    }
}

static void quant_qs4cx_f32(
    size_t n, size_t k, size_t bl, rhs_format format, const float* rhs_f32, uint8_t* rhs_qs4c32,
    uint16_t* rhs_scales_bf16) {
    if (rhs_format::nxk == format) {
        quant_nxk_qs4c32_f32(n, k, bl, rhs_f32, rhs_qs4c32, rhs_scales_bf16);
    } else {
        quant_kxn_qs4c32_f32(n, k, bl, rhs_f32, rhs_qs4c32, rhs_scales_bf16);
    }
};

static void ref_quant_qa8dx_f32(size_t m, size_t k, const float* lhs_f32, int8_t* lhs_qa8dx) {
    const size_t dst_stride = (k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t));

    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
        const float* src_ptr = lhs_f32 + row_idx * k;

        float max0 = -FLT_MAX;
        float min0 = FLT_MAX;

        // Find min/max for each channel
        for (size_t k_idx = 0; k_idx < k; ++k_idx) {
            const float src0_0 = src_ptr[k_idx];

            max0 = std::max(src0_0, max0);
            min0 = std::min(src0_0, min0);
        }

        // Maximum/minimum int8 values
        const float qmin = (float)INT8_MIN;
        const float qmax = (float)INT8_MAX;

        const float rmin0 = std::min(0.0f, min0);
        const float rmax0 = std::max(0.0f, max0);

        const float scale0 = rmin0 == rmax0 ? 1.f : (qmax - qmin) / (rmax0 - rmin0);

        // Reciprocal to quantize
        const float recip_scale0 = scale0 ? 1.0f / scale0 : 0.0f;

        const float descaled_min0 = rmin0 * scale0;
        const float descaled_max0 = rmax0 * scale0;

        const float zero_point_from_min_error0 = qmin + descaled_min0;
        const float zero_point_from_max_error0 = qmax + descaled_max0;

        float zero_point0 =
            zero_point_from_min_error0 + zero_point_from_max_error0 > 0 ? qmin - descaled_min0 : qmax - descaled_max0;

        zero_point0 = std::max(zero_point0, qmin);
        zero_point0 = std::min(zero_point0, qmax);

        // Round to nearest integer
        const int32_t nudged_zero_point0 = lrintf(zero_point0);

        int8_t* dst_ptr = (int8_t*)lhs_qa8dx + row_idx * dst_stride;

        // LHS offset at the beginning of the row
        *((float*)(dst_ptr)) = recip_scale0;
        dst_ptr += sizeof(float);
        *((int32_t*)(dst_ptr)) = -nudged_zero_point0;
        dst_ptr += sizeof(int32_t);

        // Quantize the channels
        for (size_t k_idx = 0; k_idx < k; ++k_idx) {
            const float src0_0 = src_ptr[k_idx];

            // Scale the values
            int32_t v0_s32 = (int32_t)(round(src0_0 * scale0));

            v0_s32 = v0_s32 + nudged_zero_point0;
            v0_s32 = std::max(v0_s32, INT8_MIN);
            v0_s32 = std::min(v0_s32, INT8_MAX);
            dst_ptr[0] = (int8_t)v0_s32;
            dst_ptr += sizeof(int8_t);
        }
    }
}

static void ref_matmul_mxn_mxk_nxk_f32_qa8dx_qs4c32(
    size_t m, size_t n, size_t k, size_t bl, const int8_t* lhs_qa8dx, const uint8_t* rhs_qs4c32,
    const uint16_t* scale_bf16, float* dst_f32, float scalar_min, float scalar_max) {
    const size_t num_blocks_row = get_num_blocks_per_row(k, bl);

    const size_t lhs_stride = k + sizeof(float) + sizeof(int32_t);
    const size_t rhs_stride = get_rhs_native_stride(k);

    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
        const int8_t* lhs_ptr_start = lhs_qa8dx + row_idx * lhs_stride;

        for (size_t col_idx = 0; col_idx < n; ++col_idx) {
            // Main f32 accumulator
            float main_acc = 0.0f;

            const int8_t* lhs_ptr = lhs_ptr_start;
            const uint8_t* rhs_ptr = rhs_qs4c32 + col_idx * rhs_stride;

            // Get the LHS quantization parameters stored at the
            // beginning of each row
            const float lhs_scale = *(const float*)lhs_ptr;
            lhs_ptr += sizeof(float);

            const int32_t lhs_offset = *(const int32_t*)lhs_ptr;
            lhs_ptr += sizeof(int32_t);

            for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
                const uint16_t rhs_scale_bf16 = scale_bf16[block_idx + col_idx * num_blocks_row];
                const float rhs_scale = kai_cast_f32_bf16(rhs_scale_bf16);

                int32_t iacc = 0;

                for (size_t i = 0; i < bl; ++i) {
                    const size_t k_idx = block_idx * bl + i;

                    if (k_idx >= k) {
                        break;
                    }

                    // Get the LHS values
                    const int32_t lhs_v0 = (int32_t)lhs_ptr[0];

                    // Get the RHS values
                    const uint8_t rhs_byte = rhs_ptr[0];

                    // Unpack the RHS values
                    int32_t rhs_v0 = 0;
                    if ((k_idx % 2) == 0) {
                        rhs_v0 = (((int32_t)(rhs_byte & 0x0F)) - 8);
                    } else {
                        rhs_v0 = (((int32_t)(rhs_byte >> 4)) - 8);
                    }

                    iacc += lhs_v0 * rhs_v0;
                    iacc += lhs_offset * rhs_v0;

                    lhs_ptr += 1;

                    // Increment only when k_idx is not a multiple of 2
                    rhs_ptr += k_idx % 2;
                }

                main_acc += iacc * rhs_scale;
            }

            main_acc = main_acc * lhs_scale;

            // Clamp (min-max) operation
            main_acc = std::max(main_acc, scalar_min);
            main_acc = std::min(main_acc, scalar_max);

            dst_f32[0] = main_acc;
            dst_f32 += 1;
        }
    }
};

static void ref_matmul_mxn_mxk_kxn_f32_qa8dx_qs4c32(
    size_t m, size_t n, size_t k, size_t bl, const int8_t* lhs_qa8dx, const uint8_t* rhs_qs4c32,
    const uint16_t* scale_bf16, float* dst_f32, float scalar_min, float scalar_max) {
    const size_t num_blocks_row = get_num_blocks_per_row(k, bl);

    const size_t lhs_stride = k + sizeof(float) + sizeof(int32_t);
    const size_t rhs_stride = get_rhs_native_stride(n);

    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
        const int8_t* lhs_ptr_start = lhs_qa8dx + row_idx * lhs_stride;

        for (size_t col_idx = 0; col_idx < n; ++col_idx) {
            // Main f32 accumulator
            float main_acc = 0.0f;

            const int8_t* lhs_ptr = lhs_ptr_start;
            const uint8_t* rhs_ptr = rhs_qs4c32 + (col_idx / 2);

            // Get the LHS quantization parameters stored at the
            // beginning of each row
            const float lhs_scale = *(const float*)lhs_ptr;
            lhs_ptr += sizeof(float);

            const int32_t lhs_offset = *(const int32_t*)lhs_ptr;
            lhs_ptr += sizeof(int32_t);

            for (size_t block_idx = 0; block_idx < num_blocks_row; ++block_idx) {
                const uint16_t rhs_scale_bf16 = scale_bf16[block_idx + col_idx * num_blocks_row];
                const float rhs_scale = kai_cast_f32_bf16(rhs_scale_bf16);

                int32_t iacc = 0;

                for (size_t i = 0; i < bl; ++i) {
                    const size_t k_idx = block_idx * bl + i;

                    if (k_idx >= k) {
                        break;
                    }

                    // Get the LHS values
                    const int32_t lhs_v0 = (int32_t)lhs_ptr[0];

                    // Get the RHS values
                    const uint8_t rhs_byte = rhs_ptr[0];

                    // Unpack the RHS values
                    int32_t rhs_v0 = 0;
                    if ((col_idx % 2) == 0) {
                        rhs_v0 = (((int32_t)(rhs_byte & 0x0F)) - 8);
                    } else {
                        rhs_v0 = (((int32_t)(rhs_byte >> 4)) - 8);
                    }

                    iacc += lhs_v0 * rhs_v0;
                    iacc += lhs_offset * rhs_v0;

                    lhs_ptr += 1;
                    rhs_ptr += rhs_stride;
                }

                main_acc += iacc * rhs_scale;
            }

            main_acc = main_acc * lhs_scale;

            // Clamp (min-max) operation
            main_acc = std::max(main_acc, scalar_min);
            main_acc = std::min(main_acc, scalar_max);

            dst_f32[0] = main_acc;
            dst_f32 += 1;
        }
    }
};

static void ref_matmul_f32_qa8dx_qs4c32(
    size_t m, size_t n, size_t k, size_t bl, rhs_format format, const int8_t* lhs_qa8dx, const uint8_t* rhs_qs4c32,
    const uint16_t* rhs_scales_bf16, float* dst_f32, float scalar_min, float scalar_max) {
    if (rhs_format::nxk == format) {
        ref_matmul_mxn_mxk_nxk_f32_qa8dx_qs4c32(
            m, n, k, bl, lhs_qa8dx, rhs_qs4c32, rhs_scales_bf16, dst_f32, scalar_min, scalar_max);
    } else {
        ref_matmul_mxn_mxk_kxn_f32_qa8dx_qs4c32(
            m, n, k, bl, lhs_qa8dx, rhs_qs4c32, rhs_scales_bf16, dst_f32, scalar_min, scalar_max);
    }
};

static bool is_output_correct(size_t num_rows, size_t num_cols, float tolerance, const float* ref, const float* act) {
    bool is_valid = true;

    for (size_t i = 0; i < num_rows * num_cols; ++i) {
        if (std::fabs(ref[i] - act[i]) > tolerance) {
            const size_t x = i % num_cols;
            const size_t y = i / num_cols;
            printf("ERROR![%ld][%ld]: ref=%.5f vs. act=%.5f\n", y, x, ref[i], act[i]);
            is_valid = false;
        }
    }
    return is_valid;
}

/// Compute GFLOPS from matrix dimensions and average iteration time.
/// FLOPs for matmul = 2 * M * N * K  (one multiply + one add per element)
static double compute_gflops(size_t m, size_t n, size_t k, long avg_us) {
    if (avg_us <= 0) return 0.0;
    const double flops  = 2.0 * static_cast<double>(m)
                              * static_cast<double>(n)
                              * static_cast<double>(k);
    const double time_s = static_cast<double>(avg_us) * 1e-6;
    return ((flops / time_s) / 1e9);
}

int main(int argc, char** argv) {
    const size_t num_threads = parse_thread_count(argc, argv);
    std::cout << "Using " << num_threads << " thread(s) for computations.\n";

    const size_t num_shapes = std::size(matmul_shapes);

    const size_t seed_lhs = 4568;
    const size_t seed_rhs = seed_lhs + 4;

    std::cout << "------------" << std::endl;
    for (size_t test_idx = 0; test_idx < num_shapes; ++test_idx) {
        size_t m = matmul_shapes[test_idx].m;
        size_t n = matmul_shapes[test_idx].n;
        size_t k = matmul_shapes[test_idx].k;
        size_t bl = matmul_shapes[test_idx].bl;

        std::cout << "\nTEST[" << m << ", " << n << "," << k << "] with Block Size " << bl << "\n";
        // Iterate over the RHS format (NxK or KxN)
        for (const rhs_format& format : {rhs_format::nxk, rhs_format::kxn}) {
            std::cout << "Testing RHS format = " << (format == rhs_format::nxk ? "N x K" : "K x N") << std::endl;

            const size_t lhs_native_size_f32 = m * k * sizeof(float);
            const size_t rhs_native_size_f32 = n * k * sizeof(float);
            const size_t rhs_native_size_qs4c32 =
                format == rhs_format::nxk ? n * get_rhs_native_stride(k) : k * get_rhs_native_stride(n);
            const size_t rhs_scales_size_bf16 = n * get_rhs_scale_stride(k, bl);

            // Allocate the memory
            uint8_t* lhs_native_mtx_f32 = new uint8_t[lhs_native_size_f32];
            uint8_t* rhs_native_mtx_f32 = new uint8_t[rhs_native_size_f32];
            uint8_t* rhs_native_mtx_qs4c32 = new uint8_t[rhs_native_size_qs4c32];
            uint8_t* rhs_scales_mtx_bf16 = new uint8_t[rhs_scales_size_bf16];

            fill_uniform_random(m, k, (float*)lhs_native_mtx_f32, seed_lhs);
            fill_uniform_random(n, k, (float*)rhs_native_mtx_f32, seed_rhs);

            quant_qs4cx_f32(
                n, k, bl,                          // Dimensions
                format,                            // Format (NxK or KxN)
                (const float*)rhs_native_mtx_f32,  // RHS (F32)
                rhs_native_mtx_qs4c32,             // RHS (QS4C32)
                (uint16_t*)rhs_scales_mtx_bf16);   // Scales (Bf16)

            delete[] rhs_native_mtx_f32;

            //----------- REFERENCE IMPLEMENTATION
            //------------------------------------
            //------------------------------------
            // Memory sizes for the reference implementation
            // After dynamically quantized the LHS matrix, we have the scale and offset for each
            // row. The scale (f32) and offset (int32) are stored at the beginning of each row
            const size_t lhs_ref_size_qa8dx = m * (k + sizeof(int32_t) + sizeof(float));
            const size_t dst_ref_size_f32 = m * n * sizeof(float);

            uint8_t* lhs_ref_mtx_qa8dx = new uint8_t[lhs_ref_size_qa8dx];
            uint8_t* dst_ref_mtx_f32 = new uint8_t[dst_ref_size_f32];

            ref_quant_qa8dx_f32(m, k, (const float*)lhs_native_mtx_f32, (int8_t*)lhs_ref_mtx_qa8dx);

            ref_matmul_f32_qa8dx_qs4c32(
                m, n, k,                                // Dimensions
                bl,                                     // Block length
                format,                                 // Format (NxK or KxN)
                (const int8_t*)lhs_ref_mtx_qa8dx,       // LHS
                (const uint8_t*)rhs_native_mtx_qs4c32,  // RHS
                (const uint16_t*)rhs_scales_mtx_bf16,   // Scale
                (float*)dst_ref_mtx_f32,                // DST
                -FLT_MAX, FLT_MAX);                     // Min and max for the clamp operation

            // Remove the unnecessary buffer
            delete[] lhs_ref_mtx_qa8dx;

            //----------- END REFERENCE IMPLEMENTATION
            //------------------------------------
            //------------------------------------

            //----------- MICRO-KERNELS TESTS
            //------------------------------------
            //------------------------------------
            for (size_t idx_variant = 0; idx_variant < num_ukernel_variants; ++idx_variant) {
                // Get the packing parameters
                const size_t mr = ukernel_variants[idx_variant].ukernel.get_mr();
                const size_t nr = ukernel_variants[idx_variant].ukernel.get_nr();
                const size_t kr = ukernel_variants[idx_variant].ukernel.get_kr();
                const size_t sr = ukernel_variants[idx_variant].ukernel.get_sr();

                // Get the size in bytes for the packed matrices
                const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k, mr, kr, sr);
                size_t rhs_packed_size = 0;

                if (format == rhs_format::nxk) {
                    rhs_packed_size =
                        kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(n, k, nr, kr, sr, bl, kai_dt_bf16);

                } else {
                    rhs_packed_size =
                        kai_get_rhs_packed_size_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(n, k, nr, kr, sr, bl, kai_dt_bf16);
                }

                const size_t dst_size = ukernel_variants[idx_variant].ukernel.get_dst_size(m, n);

                // Allocate the matrices
                uint8_t* lhs_packed_mtx_qa8dx = new uint8_t[lhs_packed_size];
                uint8_t* rhs_packed_mtx_qs4c32 = new uint8_t[rhs_packed_size];
                uint8_t* dst_act_mtx_f32 = new uint8_t[dst_size];

                memset(dst_act_mtx_f32, 0, dst_size);

                // If the RHS matrix contains constant values, the packing can be performed
                // only once
                if (format == rhs_format::nxk) {
                    kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params params;
                    params.lhs_zero_point = 1;
                    params.rhs_zero_point = 8;
                    params.scale_dt = kai_dt_bf16;

                    kai_run_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0(
                        1, n, k,                                  // Dimensions
                        nr, kr, sr,                               // Packing arguments
                        bl,                                       // Block length
                        (const uint8_t*)(rhs_native_mtx_qs4c32),  // RHS
                        get_rhs_native_stride(k),                 // RHS stride
                        NULL,                                     // Bias
                        rhs_scales_mtx_bf16,                      // Scale
                        get_rhs_scale_stride(k, bl),              // Scale stride
                        rhs_packed_mtx_qs4c32,                    // RHS packed
                        0, &params);

                } else {
                    kai_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0_params params;
                    params.lhs_zero_point = 1;
                    params.rhs_zero_point = 8;
                    params.scale_dt = kai_dt_bf16;

                    kai_run_rhs_pack_kxn_qsi4c32p_qsu4c32s1s0(
                        1, n, k,                                  // Dimensions
                        nr, kr, sr,                               // Packing arguments
                        bl,                                       // Block length
                        (const uint8_t*)(rhs_native_mtx_qs4c32),  // RHS
                        get_rhs_native_stride(n),                 // RHS stride
                        NULL,                                     // Bias
                        rhs_scales_mtx_bf16,                      // Scale
                        get_rhs_scale_stride(k, bl),              // Scale stride
                        rhs_packed_mtx_qs4c32,                    // RHS packed
                        0, &params);
                }

                const auto time_s = std::chrono::high_resolution_clock::now();

                // LHS packing
                kai_run_lhs_quant_pack_qai8dxp_f32(
                    m, k,                              // Dimensions
                    mr, kr, sr, 0,                     // Packing arguments
                    (const float*)lhs_native_mtx_f32,  // LHS
                    k * sizeof(float),                 // LHS stride
                    lhs_packed_mtx_qa8dx);             // LHS packed

                // Matmul
                {
                    const size_t dst_stride = n * sizeof(float);
                    const size_t lhs_offset = ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(0, k);
                    const size_t rhs_offset = ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, k, bl);
                    const size_t dst_offset = ukernel_variants[idx_variant].ukernel.get_dst_offset(0, 0, dst_stride);

                    const void* lhs_ptr = (const void*)((const char*)lhs_packed_mtx_qa8dx + lhs_offset);
                    const void* rhs_ptr = (const void*)((const char*)rhs_packed_mtx_qs4c32 + rhs_offset);
                    float* dst_ptr = (float*)((uint8_t*)dst_act_mtx_f32 + dst_offset);

                    ukernel_variants[idx_variant].ukernel.run_matmul(
                        m, n, k,           // Dimensions
                        bl,                // Block length
                        lhs_ptr,           // LHS packed
                        rhs_ptr,           // RHS packed
                        dst_ptr,           // DST
                        dst_stride,        // DST stride (row)
                        sizeof(float),     // DST stride (col)
                        -FLT_MAX, FLT_MAX  // Min and max for the clamp operation
                    );
                }

                const auto time_e = std::chrono::high_resolution_clock::now();

                const auto elap = std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s);

                const bool is_valid =
                    is_output_correct(m, n, 0.0001f, (const float*)dst_ref_mtx_f32, (const float*)dst_act_mtx_f32);

                std::cout << "TEST[" << idx_variant << "]: Dynamic quantization + matmul" << std::endl;
                std::cout << "- ukernel: " << ukernel_variants[idx_variant].name << std::endl;
                if (is_valid) {
                    std::cout << "- Status: PASSED" << std::endl;
                    std::cout << "- Performance: " << elap.count() << " us" << std::endl;
                } else {
                    std::cout << "Status: FAILED" << std::endl;
                }
                std::cout << "------------" << std::endl;
                delete[] lhs_packed_mtx_qa8dx;
                delete[] rhs_packed_mtx_qs4c32;
                delete[] dst_act_mtx_f32;
            }
            delete[] lhs_native_mtx_f32;
            delete[] rhs_native_mtx_qs4c32;
            delete[] rhs_scales_mtx_bf16;
            delete[] dst_ref_mtx_f32;
        }
    }
    // ── QMX MOPA kernel test (multi-threaded, 16 iterations) ─────────────────
    {
        const size_t m_qmx = 512;
        const size_t n_qmx = 512;
        const size_t k_qmx = 512;
        const size_t bl_qmx = 32;

        std::cout << "\nTEST QMX MOPA [" << m_qmx << ", " << n_qmx << ", " << k_qmx
                  << "] with Block Size " << bl_qmx << "\n";
        std::cout << "Testing RHS format = N x K\n";

        const size_t lhs_native_size_f32_qmx = m_qmx * k_qmx * sizeof(float);
        const size_t rhs_native_size_f32_qmx = n_qmx * k_qmx * sizeof(float);
        const size_t rhs_native_size_qs4c32_qmx = n_qmx * get_rhs_native_stride(k_qmx);
        const size_t rhs_scales_size_bf16_qmx = n_qmx * get_rhs_scale_stride(k_qmx, bl_qmx);

        uint8_t* lhs_native_mtx_f32_qmx = new uint8_t[lhs_native_size_f32_qmx];
        uint8_t* rhs_native_mtx_f32_qmx = new uint8_t[rhs_native_size_f32_qmx];
        uint8_t* rhs_native_mtx_qs4c32_qmx = new uint8_t[rhs_native_size_qs4c32_qmx];
        uint8_t* rhs_scales_mtx_bf16_qmx = new uint8_t[rhs_scales_size_bf16_qmx];

        fill_uniform_random(m_qmx, k_qmx, (float*)lhs_native_mtx_f32_qmx, seed_lhs);
        fill_uniform_random(n_qmx, k_qmx, (float*)rhs_native_mtx_f32_qmx, seed_rhs);

        quant_qs4cx_f32(
            n_qmx, k_qmx, bl_qmx,
            rhs_format::nxk,
            (const float*)rhs_native_mtx_f32_qmx,
            rhs_native_mtx_qs4c32_qmx,
            (uint16_t*)rhs_scales_mtx_bf16_qmx);

        delete[] rhs_native_mtx_f32_qmx;

        // Reference implementation
        const size_t lhs_ref_size_qa8dx_qmx = m_qmx * (k_qmx + sizeof(int32_t) + sizeof(float));
        const size_t dst_ref_size_f32_qmx = m_qmx * n_qmx * sizeof(float);

        uint8_t* lhs_ref_mtx_qa8dx_qmx = new uint8_t[lhs_ref_size_qa8dx_qmx];
        uint8_t* dst_ref_mtx_f32_qmx = new uint8_t[dst_ref_size_f32_qmx];

        ref_quant_qa8dx_f32(m_qmx, k_qmx, (const float*)lhs_native_mtx_f32_qmx,
                            (int8_t*)lhs_ref_mtx_qa8dx_qmx);

        ref_matmul_f32_qa8dx_qs4c32(
            m_qmx, n_qmx, k_qmx, bl_qmx,
            rhs_format::nxk,
            (const int8_t*)lhs_ref_mtx_qa8dx_qmx,
            (const uint8_t*)rhs_native_mtx_qs4c32_qmx,
            (const uint16_t*)rhs_scales_mtx_bf16_qmx,
            (float*)dst_ref_mtx_f32_qmx,
            -FLT_MAX, FLT_MAX);

        delete[] lhs_ref_mtx_qa8dx_qmx;

        for (size_t idx_variant = 0; idx_variant < num_qmx_ukernel_variants; ++idx_variant) {
            std::cout << "Testing " << qmx_ukernel_variants[idx_variant].name << "\n";

            const size_t mr = qmx_ukernel_variants[idx_variant].ukernel.get_mr();
            const size_t nr = qmx_ukernel_variants[idx_variant].ukernel.get_nr();
            const size_t kr = qmx_ukernel_variants[idx_variant].ukernel.get_kr();
            const size_t sr = qmx_ukernel_variants[idx_variant].ukernel.get_sr();

            const size_t lhs_packed_size_qmx =
                kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m_qmx, k_qmx, mr, kr, sr);

            kai_rhs_pack_nxk_qsi4c32p_qsu4c32s1s0_params rhs_params;
            rhs_params.lhs_zero_point = 1;
            rhs_params.rhs_zero_point = 8;
            rhs_params.scale_dt = kai_dt_bf16;

            const size_t rhs_packed_size_qmx =
                kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
                    n_qmx, k_qmx, nr, kr, sr, bl_qmx, kai_dt_bf16);
            const size_t dst_size_qmx =
                qmx_ukernel_variants[idx_variant].ukernel.get_dst_size(m_qmx, n_qmx);

            uint8_t* lhs_packed_mtx_qa8dx_qmx = new uint8_t[lhs_packed_size_qmx];
            uint8_t* rhs_packed_mtx_qs4c32_qmx = new uint8_t[rhs_packed_size_qmx];
            uint8_t* dst_act_mtx_f32_qmx = new uint8_t[dst_size_qmx];

            // Pack RHS once (constant weights, done before threading)
            kai_run_rhs_pack_nxk_qsi4c32ps1s0nrx4_qsu4c32s1s0_neon(
                1, n_qmx, k_qmx,
                nr, kr, sr, bl_qmx,
                (const uint8_t*)rhs_native_mtx_qs4c32_qmx,
                get_rhs_native_stride(k_qmx),
                NULL,
                rhs_scales_mtx_bf16_qmx,
                get_rhs_scale_stride(k_qmx, bl_qmx),
                rhs_packed_mtx_qs4c32_qmx,
                0, &rhs_params);

            // ── Phase 1: pack LHS once (not timed) ───────────────────────────
            auto lhs_pack_worker = [&](int thread_index) {
                const size_t m_step =
                    qmx_ukernel_variants[idx_variant].ukernel.get_m_step();
                const size_t num_m_per_thread =
                    roundup(m_qmx, m_step * num_threads) / num_threads;
                const size_t m_start = static_cast<size_t>(thread_index) * num_m_per_thread;
                if (m_start >= m_qmx) return;
                const size_t m_to_process = std::min(num_m_per_thread, m_qmx - m_start);

                const size_t lhs_src_offset =
                    kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(m_start, k_qmx * sizeof(float));
                const float* src_ptr =
                    (const float*)lhs_native_mtx_f32_qmx + lhs_src_offset / sizeof(float);
                const size_t lhs_packed_offset =
                    qmx_ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, k_qmx);
                void* lhs_packed_ptr = lhs_packed_mtx_qa8dx_qmx + lhs_packed_offset;

                kai_run_lhs_quant_pack_qai8dxp_f32(
                    m_to_process, k_qmx, mr, kr, sr, 0,
                    src_ptr, k_qmx * sizeof(float),
                    lhs_packed_ptr);
            };

            {
                std::vector<std::thread> pack_threads;
                pack_threads.reserve(num_threads);
                for (size_t i = 0; i < num_threads; ++i)
                    pack_threads.emplace_back(lhs_pack_worker, static_cast<int>(i));
                for (auto& t : pack_threads) t.join();
            }

            // ── Phase 2: run matmul 60000 times and measure total elapsed time ──
            constexpr int num_iterations = 60000;

            auto matmul_worker = [&](int thread_index) {
                const size_t m_step =
                    qmx_ukernel_variants[idx_variant].ukernel.get_m_step();
                const size_t num_m_per_thread =
                    roundup(m_qmx, m_step * num_threads) / num_threads;
                const size_t m_start = static_cast<size_t>(thread_index) * num_m_per_thread;
                if (m_start >= m_qmx) return;
                const size_t m_to_process = std::min(num_m_per_thread, m_qmx - m_start);

                const size_t dst_stride = n_qmx * sizeof(float);
                const size_t lhs_packed_offset =
                    qmx_ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, k_qmx);
                const void* lhs_packed_ptr = lhs_packed_mtx_qa8dx_qmx + lhs_packed_offset;
                const size_t rhs_packed_offset =
                    qmx_ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, k_qmx, bl_qmx);
                const void* rhs_packed_ptr = rhs_packed_mtx_qs4c32_qmx + rhs_packed_offset;
                const size_t dst_offset =
                    qmx_ukernel_variants[idx_variant].ukernel.get_dst_offset(m_start, 0, dst_stride);
                float* dst_ptr = (float*)(dst_act_mtx_f32_qmx + dst_offset);

                // Each thread runs run_matmul num_iterations times.
                // join() in the main thread blocks until this loop completes,
                // so time_e - time_s covers all num_iterations calls.
                for (int iter = 0; iter < num_iterations; ++iter) {
                    qmx_ukernel_variants[idx_variant].ukernel.run_matmul(
                        m_to_process, n_qmx, k_qmx, bl_qmx,
                        lhs_packed_ptr, rhs_packed_ptr,
                        dst_ptr, dst_stride, sizeof(float),
                        -FLT_MAX, FLT_MAX);
                }
            };

            //Start Time
            const auto time_s = std::chrono::high_resolution_clock::now();

            {
                std::vector<std::thread> threads;
                threads.reserve(num_threads);
                // Create worker threads; each runs run_matmul num_iterations times
                for (size_t i = 0; i < num_threads; ++i)
                    threads.emplace_back(matmul_worker, static_cast<int>(i));
                // join() blocks until every thread has finished all num_iterations calls
                for (auto& t : threads) t.join();
            }

            //End Time
            const auto time_e = std::chrono::high_resolution_clock::now();
            const auto total_us =
                std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s).count();
            const long avg_us = total_us / num_iterations;

            const bool is_valid =
                is_output_correct(m_qmx, n_qmx, 0.0001f,
                                  (const float*)dst_ref_mtx_f32_qmx,
                                  (const float*)dst_act_mtx_f32_qmx);

            std::cout << "TEST[" << idx_variant << "]: Dynamic quantization + matmul\n";
            std::cout << "- ukernel: " << qmx_ukernel_variants[idx_variant].name << "\n";
            if (is_valid) {
                const double gflops = compute_gflops(m_qmx, n_qmx, k_qmx, avg_us);
                std::cout << "- Status: PASSED\n";
                std::cout << "- Iterations: " << num_iterations << "\n";
                std::cout << "- Total Performance time: " << total_us << " us\n";
                std::cout << "- Avg Performance time per iteration: " << avg_us << " us\n";
                std::cout << std::fixed << std::setprecision(2) << "- GFLOPS: " << gflops << "\n";
            } else {
                std::cout << "- Status: FAILED\n";
            }
            std::cout << "------------\n";

            delete[] lhs_packed_mtx_qa8dx_qmx;
            delete[] rhs_packed_mtx_qs4c32_qmx;
            delete[] dst_act_mtx_f32_qmx;
        }

        delete[] lhs_native_mtx_f32_qmx;
        delete[] rhs_native_mtx_qs4c32_qmx;
        delete[] rhs_scales_mtx_bf16_qmx;
        delete[] dst_ref_mtx_f32_qmx;
    }
}

//----------- END MICRO-KERNELS TESTS
//------------------------------------
//------------------------------------

#endif  // Architectural feature check
