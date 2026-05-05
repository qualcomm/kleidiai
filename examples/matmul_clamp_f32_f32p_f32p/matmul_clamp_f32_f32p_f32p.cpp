//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//

// Example usage for matrix multiplication of two single-precision floating-point (FP32) matrices
// with FP32 output and an optional bias vector.
//
// Tests one micro-kernel variant:
//   TEST[0]: kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa
//            (QMX MOPA, packed LHS + packed RHS+bias, multi-threaded, 40000 iterations)
// Multi-threading is supported via --threads <N>.

#if !defined(__aarch64__)
#error This file must be compiled for AArch64.
#else
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

// KleidiAI common utilities (kai_roundup, etc.)
#include "kai/kai_common.h"

// QMX MOPA kernel – packed LHS and packed RHS+bias
#include "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa.h"
#include "kai_matmul_clamp_f32_f32p_f32p_interface.h"
#include "kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"

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

// ─── QMX MOPA ukernel variant table ─────────────────────────────────────────

struct kai_matmul_ukernel_f32_f32p_f32p {
    kai_matmul_clamp_f32_f32p_f32p_ukernel ukernel;
    std::string name = {};
};

static kai_matmul_ukernel_f32_f32p_f32p qmx_ukernel_variants[] = {
    {
        {
            kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa,
            kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa,
            kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa,
            kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa,
            kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa,
            kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa,
            kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa,
            kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa,
            kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa,
            kai_get_dst_size_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa,
            kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa,
        },
        "matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_qmx_mopa",
    },
};

static const size_t num_qmx_ukernel_variants =
    sizeof(qmx_ukernel_variants) / sizeof(qmx_ukernel_variants[0]);

// ─── Helper functions ────────────────────────────────────────────────────────

/// Fills the matrix with incremental values scaled by weight
static void fill_matrix(size_t num_rows, size_t num_cols, float* dst, float weight) {
    for (size_t i = 0; i < num_rows * num_cols; i++) {
        dst[i] = float((i + 1) * weight);
    }
}

/// Reference scalar FP32 matrix multiplication with bias and clamp.
/// RHS is stored in KxN layout (K rows, N columns).
static void run_matmul_ref(
    size_t m, size_t n, size_t k,
    const float* lhs,   // MxK
    const float* rhs,   // KxN
    const float* bias,  // N
    float* dst,         // MxN
    float scalar_min, float scalar_max) {
    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
        for (size_t col_idx = 0; col_idx < n; ++col_idx) {
            float acc = bias ? bias[col_idx] : 0.0f;
            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                acc += lhs[row_idx * k + k_idx] * rhs[col_idx + n * k_idx];
            }
            acc = std::max(acc, scalar_min);
            acc = std::min(acc, scalar_max);
            dst[row_idx * n + col_idx] = acc;
        }
    }
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

/// Verify the micro-kernel output matches the reference implementation.
/// Uses relative tolerance: |ref - act| <= tolerance * max(|ref|, 1.0)
static bool is_output_correct(
    size_t num_rows, size_t num_cols, float tolerance,
    const float* ref, const float* act) {
    bool is_valid = true;
    for (size_t i = 0; i < num_rows * num_cols; ++i) {
        const float abs_diff = std::fabs(ref[i] - act[i]);
        const float threshold = tolerance * std::max(std::fabs(ref[i]), 1.0f);
        if (abs_diff > threshold) {
            const size_t x = i % num_cols;
            const size_t y = i / num_cols;
            std::cout << std::setprecision(5) << std::fixed
                      << "ERROR![" << y << "][" << x << "]: ref=" << ref[i]
                      << " vs. act=" << act[i] << "\n";
            is_valid = false;
        }
    }
    return is_valid;
}

// ─── main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int ret = 0;

    const size_t num_threads = parse_thread_count(argc, argv);
    std::cout << "Using " << num_threads << " thread(s) for computations.\n";

    // Matrix dimensions
    const size_t M = 512;  // LHS rows / output rows
    const size_t N = 512;  // RHS columns / output columns
    const size_t K = 512;  // Common dimension

    std::cout << "Matrix dimensions: M=" << M << " N=" << N << " K=" << K << "\n\n";

    const size_t lhs_size  = M * K;
    const size_t rhs_size  = K * N;  // KxN layout
    const size_t bias_size = N;
    const size_t dst_size  = M * N;

    // Allocate and fill input matrices
    float* lhs  = new float[lhs_size];
    float* rhs  = new float[rhs_size];
    float* bias = new float[bias_size];

    fill_matrix(M, K, lhs,  0.001f);
    fill_matrix(K, N, rhs,  0.001f);
    fill_matrix(1, N, bias, 0.1f);

    // ── Reference implementation ──────────────────────────────────────────────
    float* dst_ref = new float[dst_size];
    run_matmul_ref(M, N, K, lhs, rhs, bias, dst_ref, -FLT_MAX, FLT_MAX);

    // ── QMX MOPA kernel (packed LHS + packed RHS+bias, multi-threaded) ────────
    for (size_t idx_variant = 0; idx_variant < num_qmx_ukernel_variants; ++idx_variant) {
        std::cout << "Testing " << qmx_ukernel_variants[idx_variant].name << "\n";

        const size_t mr = qmx_ukernel_variants[idx_variant].ukernel.get_mr();
        const size_t nr = qmx_ukernel_variants[idx_variant].ukernel.get_nr();
        const size_t kr = qmx_ukernel_variants[idx_variant].ukernel.get_kr();
        const size_t sr = qmx_ukernel_variants[idx_variant].ukernel.get_sr();

        // Compute packed buffer sizes
        const size_t lhs_packed_size =
            kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(M, K, mr, kr, sr);
        const size_t rhs_packed_size =
            kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(N, K);
        const size_t dst_size_bytes =
            qmx_ukernel_variants[idx_variant].ukernel.get_dst_size(M, N);

        uint8_t* lhs_packed = new uint8_t[lhs_packed_size];
        uint8_t* rhs_packed = new uint8_t[rhs_packed_size];
        float*   dst        = new float[dst_size_bytes / sizeof(float)];

        // Pack RHS + bias once (constant weights, done before threading)
        const size_t rhs_stride_row = N * sizeof(float);
        kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
            1, N, K, nr, kr, sr,
            rhs_stride_row, rhs, bias,
            /*scale=*/NULL, rhs_packed,
            /*extra_bytes=*/0, /*params=*/NULL);

        // ── Phase 1: pack LHS once (not timed) ───────────────────────────────
        auto lhs_pack_worker = [&](int thread_index) {
            const size_t m_step =
                qmx_ukernel_variants[idx_variant].ukernel.get_m_step();
            const size_t num_m_per_thread =
                kai_roundup(M, m_step * num_threads) / num_threads;
            const size_t m_start = static_cast<size_t>(thread_index) * num_m_per_thread;
            if (m_start >= M) return;
            const size_t m_to_process = std::min(num_m_per_thread, M - m_start);

            const size_t lhs_src_offset =
                kai_get_lhs_offset_lhs_pack_f32p2vlx1_f32_sme(
                    m_start, K * sizeof(float));
            const void* lhs_src_ptr =
                reinterpret_cast<const uint8_t*>(lhs) + lhs_src_offset;

            const size_t lhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(
                    m_start, K);
            void* lhs_packed_ptr = lhs_packed + lhs_packed_offset;

            kai_run_lhs_pack_f32p2vlx1_f32_sme(
                m_to_process, K, mr, kr, sr,
                /*m_idx_start=*/0,
                lhs_src_ptr, K * sizeof(float),
                lhs_packed_ptr);
        };

        {
            std::vector<std::thread> pack_threads;
            pack_threads.reserve(num_threads);
            for (size_t i = 0; i < num_threads; ++i)
                pack_threads.emplace_back(lhs_pack_worker, static_cast<int>(i));
            for (auto& t : pack_threads) t.join();
        }

        // ── Phase 2: run matmul 60000 times and measure total elapsed time ───────
        constexpr int num_iterations = 60000;

        auto matmul_worker = [&](int thread_index) {
            const size_t m_step =
                qmx_ukernel_variants[idx_variant].ukernel.get_m_step();
            const size_t num_m_per_thread =
                kai_roundup(M, m_step * num_threads) / num_threads;
            const size_t m_start = static_cast<size_t>(thread_index) * num_m_per_thread;
            if (m_start >= M) return;
            const size_t m_to_process = std::min(num_m_per_thread, M - m_start);

            const size_t dst_stride_row = N * sizeof(float);

            const size_t lhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, K);
            const void* lhs_packed_ptr = lhs_packed + lhs_packed_offset;

            const size_t rhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, K);
            const void* rhs_packed_ptr = rhs_packed + rhs_packed_offset;

            const size_t dst_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_dst_offset(
                    m_start, 0, dst_stride_row);
            void* dst_ptr = reinterpret_cast<uint8_t*>(dst) + dst_offset;

            // Each thread runs run_matmul num_iterations times.
            // join() in the main thread blocks until this loop completes,
            // so time_e - time_s covers all num_iterations calls.
            for (int iter = 0; iter < num_iterations; ++iter) {
                qmx_ukernel_variants[idx_variant].ukernel.run_matmul(
                    m_to_process, N, K,
                    lhs_packed_ptr, rhs_packed_ptr,
                    dst_ptr,
                    dst_stride_row, sizeof(float),
                    -FLT_MAX, FLT_MAX);
            }
        };


        //Start Time
        const auto time_s = std::chrono::high_resolution_clock::now();
        //const auto time_s = std::chrono::steady_clock::now();        

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
        //const auto time_e = std::chrono::steady_clock::now();        
        const auto total_us =
            std::chrono::duration_cast<std::chrono::microseconds>(time_e - time_s).count();
        const long avg_us = total_us / num_iterations;

        // 0.1% relative tolerance
        const bool is_valid =
            is_output_correct(M, N, 0.001f, dst_ref, dst);

        const double gflops = compute_gflops(M, N, K, avg_us);

        printf("TEST[%zu] = %s\n", idx_variant, is_valid ? "PASSED" : "FAILED");
        std::cout << "- ukernel: " << qmx_ukernel_variants[idx_variant].name << "\n";
        std::cout << "- Iterations: " << num_iterations << "\n";
        std::cout << "- Total Performance time: " << total_us << " us\n";
        std::cout << "- Avg Performance time per iteration: " << avg_us << " us\n";
        std::cout << std::fixed << std::setprecision(2)
                  << "- GFLOPS: " << gflops << "\n\n";
        if (!is_valid) ret = 1;

        delete[] lhs_packed;
        delete[] rhs_packed;
        delete[] dst;
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    delete[] lhs;
    delete[] rhs;
    delete[] bias;
    delete[] dst_ref;

    return ret;
}

#endif  // Architectural features check.
