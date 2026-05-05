//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//

// Example usage for matrix multiplication with packed int8 asymmetric LHS (qai8p)
// and int8 symmetric per-channel RHS with scale+bias (qsi8cxpsb), producing int8 output.
//
// Tests one QMX MOPA micro-kernel variant:
//   TEST[0]: kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa
//            (QMX MOPA, packed LHS + packed RHS, multi-threaded, 10000 iterations)
//
// Multi-threading is supported via --threads <N>.

#if !defined(__aarch64__)
#error This file must be compiled for AArch64.
#else
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "kai/kai_common.h"
#include "kai_lhs_pack_x8p2vlx4_x8_sme.h"
#include "kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa.h"
#include "kai_matmul_clamp_qai8_qai8p_qsi8cxpsb_interface.h"
#include "kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.h"

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
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(EXIT_SUCCESS); }
        if (arg == "--threads" || arg == "-t") {
            if ((i + 1) >= argc) { std::cerr << "--threads expects a value\n"; print_usage(argv[0]); std::exit(EXIT_FAILURE); }
            thread_count = parse_thread_count_value(argv[++i]);
            continue;
        }
        const std::string prefix = "--threads=";
        if (arg.rfind(prefix, 0) == 0) { thread_count = parse_thread_count_value(arg.substr(prefix.size())); continue; }
        std::cerr << "Unrecognized argument: " << arg << "\n"; print_usage(argv[0]); std::exit(EXIT_FAILURE);
    }
    return thread_count;
}

// ─── QMX MOPA ukernel variant table ─────────────────────────────────────────

struct kai_matmul_ukernel_qai8_qa8p_qs8cxpsb_qmx {
    kai_matmul_clamp_qai8_qai8p_qsi8cxpsb_ukernel ukernel;
    std::string name = {};
};

static kai_matmul_ukernel_qai8_qa8p_qs8cxpsb_qmx qmx_ukernel_variants[] = {
    {{kai_get_m_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa,
      kai_get_n_step_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa,
      kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa,
      kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa,
      kai_get_kr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa,
      kai_get_sr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa,
      kai_get_lhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa,
      kai_get_rhs_packed_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa,
      kai_get_dst_offset_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa,
      kai_get_dst_size_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa,
      kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa},
     "matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa"},
};

static const size_t num_qmx_ukernel_variants =
    sizeof(qmx_ukernel_variants) / sizeof(qmx_ukernel_variants[0]);

// ─── Helper functions ─────────────────────────────────────────────────────────

static void fill_uniform_random_i8(size_t count, int8_t* dst, size_t seed, int range = 10) {
    std::srand(seed);
    for (size_t i = 0; i < count; i++) {
        dst[i] = (int8_t)((std::rand() % (2 * range + 1)) - range);
    }
}

/// Reference scalar matmul: LHS=int8 (MxK), RHS=int8 (KxN), output=int8
/// Uses scale_multiplier for requantization.
static void ref_matmul_qai8_qa8_qs8cx(
    size_t m, size_t n, size_t k,
    const int8_t* lhs,       // MxK
    const int8_t* rhs,       // KxN
    const float* rhs_scales, // per-channel scale (N)
    const float* bias,       // per-channel bias (N), may be NULL
    int8_t* dst,             // MxN
    float scale_multiplier,
    const struct kai_matmul_requantize32_params* params) {
    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
        for (size_t n_idx = 0; n_idx < n; ++n_idx) {
            int32_t acc = 0;
            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                acc += (int32_t)lhs[m_idx * k + k_idx] * (int32_t)rhs[k_idx * n + n_idx];
            }
            float result = (float)acc * rhs_scales[n_idx] * scale_multiplier;
            if (bias) result += bias[n_idx];
            int32_t q = (int32_t)std::round(result) + params->output_zero_point;
            q = std::max(q, params->min_value);
            q = std::min(q, params->max_value);
            dst[m_idx * n + n_idx] = (int8_t)q;
        }
    }
}

static bool is_output_correct_i8(size_t num_rows, size_t num_cols,
                                  const int8_t* ref, const int8_t* act) {
    bool is_valid = true;
    for (size_t i = 0; i < num_rows * num_cols; ++i) {
        if (ref[i] != act[i]) {
            const size_t x = i % num_cols, y = i / num_cols;
            printf("ERROR![%zu][%zu]: ref=%d vs. act=%d\n", y, x, (int)ref[i], (int)act[i]);
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

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int ret = 0;

    const size_t num_threads = parse_thread_count(argc, argv);
    std::cout << "Using " << num_threads << " thread(s) for computations.\n";

    // Matrix dimensions
    const size_t M = 512;
    const size_t N = 512;
    const size_t K = 512;

    std::cout << "Matrix dimensions: M=" << M << " N=" << N << " K=" << K << "\n\n";

    const size_t seed_lhs = 4568;
    const size_t seed_rhs = seed_lhs + 4;

    // Allocate and fill input matrices (int8)
    int8_t* lhs = new int8_t[M * K];
    int8_t* rhs = new int8_t[K * N];  // KxN layout

    fill_uniform_random_i8(M * K, lhs, seed_lhs, 10);
    fill_uniform_random_i8(K * N, rhs, seed_rhs, 10);

    // Per-channel RHS scales and bias
    float* rhs_scales = new float[N];
    float* rhs_bias   = new float[N];
    for (size_t n_idx = 0; n_idx < N; ++n_idx) {
        rhs_scales[n_idx] = 1.0f / 127.0f;
        rhs_bias[n_idx]   = 0.0f;
    }

    // Requantization parameters
    // scale_multiplier = lhs_scale * rhs_scale / output_scale
    // For simplicity: lhs_scale = 1/127, rhs_scale = 1/127, output_scale = 1/127
    // => scale_multiplier = 1/127
    const float scale_multiplier = 1.0f / 127.0f;

    struct kai_matmul_requantize32_params req_params;
    req_params.min_value         = -128;
    req_params.max_value         = 127;
    req_params.output_zero_point = 0;

    // Reference implementation
    int8_t* dst_ref = new int8_t[M * N];
    ref_matmul_qai8_qa8_qs8cx(M, N, K, lhs, rhs, rhs_scales, rhs_bias,
                               dst_ref, scale_multiplier, &req_params);

    // ── QMX MOPA kernel (packed LHS + packed RHS, multi-threaded) ─────────────
    for (size_t idx_variant = 0; idx_variant < num_qmx_ukernel_variants; ++idx_variant) {
        std::cout << "Testing " << qmx_ukernel_variants[idx_variant].name << "\n";

        const size_t mr = qmx_ukernel_variants[idx_variant].ukernel.get_mr();
        const size_t nr = qmx_ukernel_variants[idx_variant].ukernel.get_nr();
        const size_t kr = qmx_ukernel_variants[idx_variant].ukernel.get_kr();
        const size_t sr = qmx_ukernel_variants[idx_variant].ukernel.get_sr();

        const size_t lhs_packed_size =
            kai_get_lhs_packed_size_lhs_pack_x8p2vlx4_x8_sme(M, K, mr, kr, sr);
        const size_t rhs_packed_size =
            kai_get_rhs_packed_size_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(N, K);
        const size_t dst_size_bytes =
            qmx_ukernel_variants[idx_variant].ukernel.get_dst_size(M, N);

        uint8_t* lhs_packed = new uint8_t[lhs_packed_size];
        uint8_t* rhs_packed = new uint8_t[rhs_packed_size];
        int8_t*  dst        = new int8_t[dst_size_bytes];

        // Pack RHS once (constant weights, done before threading)
        struct kai_rhs_pack_qsi8cx_params rhs_pack_params;
        rhs_pack_params.lhs_zero_point  = 0;
        rhs_pack_params.scale_multiplier = scale_multiplier;

        kai_run_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme(
            1, N, K, nr, kr, sr,
            N * sizeof(int8_t),  // rhs_stride_row (KxN: stride = N)
            rhs,
            rhs_bias,
            rhs_scales,
            rhs_packed,
            /*extra_bytes=*/0,
            &rhs_pack_params);

        // ── Phase 1: pack LHS once (not timed) ───────────────────────────────
        auto lhs_pack_worker = [&](int thread_index) {
            const size_t m_step = qmx_ukernel_variants[idx_variant].ukernel.get_m_step();
            const size_t num_m_per_thread = kai_roundup(M, m_step * num_threads) / num_threads;
            const size_t m_start = static_cast<size_t>(thread_index) * num_m_per_thread;
            if (m_start >= M) return;
            const size_t m_to_process = std::min(num_m_per_thread, M - m_start);

            const size_t lhs_src_offset =
                kai_get_lhs_offset_lhs_pack_x8p2vlx4_x8_sme(m_start, K * sizeof(int8_t));
            const void* lhs_src_ptr =
                reinterpret_cast<const uint8_t*>(lhs) + lhs_src_offset;
            const size_t lhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, K);
            void* lhs_packed_ptr = lhs_packed + lhs_packed_offset;

            kai_run_lhs_pack_x8p2vlx4_x8_sme(
                m_to_process, K, mr, kr, sr,
                /*m_idx_start=*/0,
                lhs_src_ptr, K * sizeof(int8_t),
                lhs_packed_ptr);
        };

        {
            std::vector<std::thread> pack_threads;
            pack_threads.reserve(num_threads);
            for (size_t i = 0; i < num_threads; ++i)
                pack_threads.emplace_back(lhs_pack_worker, static_cast<int>(i));
            for (auto& t : pack_threads) t.join();
        }

        // ── Phase 2: run matmul 60000 times and measure total elapsed time ────
        constexpr int num_iterations = 60000;

        auto matmul_worker = [&](int thread_index) {
            const size_t m_step = qmx_ukernel_variants[idx_variant].ukernel.get_m_step();
            const size_t num_m_per_thread = kai_roundup(M, m_step * num_threads) / num_threads;
            const size_t m_start = static_cast<size_t>(thread_index) * num_m_per_thread;
            if (m_start >= M) return;
            const size_t m_to_process = std::min(num_m_per_thread, M - m_start);

            const size_t dst_stride_row = N * sizeof(int8_t);
            const size_t lhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, K);
            const void* lhs_packed_ptr = lhs_packed + lhs_packed_offset;
            const size_t rhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, K);
            const void* rhs_packed_ptr = rhs_packed + rhs_packed_offset;
            const size_t dst_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_dst_offset(m_start, 0, dst_stride_row);
            void* dst_ptr = reinterpret_cast<uint8_t*>(dst) + dst_offset;

            for (int iter = 0; iter < num_iterations; ++iter) {
                qmx_ukernel_variants[idx_variant].ukernel.run_matmul(
                    m_to_process, N, K,
                    lhs_packed_ptr, rhs_packed_ptr,
                    dst_ptr, dst_stride_row, sizeof(int8_t),
                    &req_params);
            }
        };

        //Start Time
        const auto time_s = std::chrono::high_resolution_clock::now();

        {
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            for (size_t i = 0; i < num_threads; ++i)
                threads.emplace_back(matmul_worker, static_cast<int>(i));
            for (auto& t : threads) t.join();
        }

        //End Time
        const auto time_e = std::chrono::high_resolution_clock::now();
        const auto total_us =
            std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s).count();
        const long avg_us = total_us / num_iterations;

        const bool is_valid = is_output_correct_i8(M, N, dst_ref, dst);

        printf("TEST[%zu] = %s\n", idx_variant, is_valid ? "PASSED" : "FAILED");
        std::cout << "- ukernel: " << qmx_ukernel_variants[idx_variant].name << "\n";
        std::cout << "- Iterations: " << num_iterations << "\n";
        std::cout << "- Total Performance time: " << total_us << " us\n";
        const double gflops = compute_gflops(M, N, K, avg_us);
        std::cout << "- Avg Performance time per iteration: " << avg_us << " us\n";
        std::cout << std::fixed << std::setprecision(2) << "- GFLOPS: " << gflops << "\n\n";
        if (!is_valid) ret = 1;

        delete[] lhs_packed;
        delete[] rhs_packed;
        delete[] dst;
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    delete[] lhs;
    delete[] rhs;
    delete[] rhs_scales;
    delete[] rhs_bias;
    delete[] dst_ref;

    return ret;
}

#endif  // Architectural features check.
