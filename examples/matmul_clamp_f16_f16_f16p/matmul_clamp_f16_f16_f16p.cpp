//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Example usage for matrix multiplication of two half precision floating-point (FP16) matrices and the accumulation of
// the result into an FP16 destination matrix.
//
// The activations and the weights, stored in the LHS and RHS matrices respectively, are both non-transposed matrices.
// The matrix multiplication computation is performed using floating-point fused multiply-add to accumulator (FMLA)
// vector instructions present in the FEAT_FP16 Arm® architecture feature.
//
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_FP16.
#else
#include <arm_neon.h>

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

// Include micro-kernel variants
#include "kai/kai_common.h"
#include "kai_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla.h"
#include "kai_matmul_clamp_f16_f16_f16p_interface.h"
#include "kai_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon.h"

// QMX MOPA kernel – packed LHS and packed RHS+bias
#include "kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa.h"
#include "kai_matmul_clamp_f16_f16p_f16p_interface.h"
#include "kai_lhs_pack_x16p2vlx2_x16_sme.h"
#include "kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme.h"

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

namespace {
/// Micro-kernel interface
constexpr kai_matmul_clamp_f16_f16_f16p_ukernel ukernel{
    kai_get_m_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_n_step_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_nr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_kr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_sr_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_lhs_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_rhs_packed_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_dst_offset_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_get_dst_size_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla,
    kai_run_matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla};

/// Reference implementation of matrix multiplication
void run_matmul_ref(
    size_t m, size_t n, size_t k, const float16_t* lhs, const float16_t* rhs, const float16_t* bias, float16_t* dst,
    float scalar_min, float scalar_max) {
    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
        for (size_t col_idx = 0; col_idx < n; ++col_idx) {
            float16_t acc = bias[col_idx];

            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                acc += lhs[row_idx * k + k_idx] * rhs[col_idx + n * k_idx];
            }
            acc = std::max(acc, static_cast<float16_t>(scalar_min));
            acc = std::min(acc, static_cast<float16_t>(scalar_max));

            dst[row_idx * n + col_idx] = acc;
        }
    }
}

/// Fills the matrix with incremental values
void fill_matrix(size_t num_rows, size_t num_cols, float16_t* dst, const float16_t weight) {
    for (size_t i = 0; i < num_rows * num_cols; i++) {
        dst[i] = float16_t(i * weight);
    }
}

/// Print the matrix
void print_matrix(size_t num_rows, size_t num_cols, const char* name, const float16_t* src) {
    std::cout << name << " = [\n";
    for (size_t y = 0; y < num_rows; ++y) {
        std::cout << "  [";
        for (size_t x = 0; x < num_cols; ++x) {
            std::cout << std::setprecision(2) << std::fixed << src[y * num_cols + x] << ", ";
        }
        std::cout << ("],\n");
    }
    std::cout << ("]\n\n");
}

/// Verify the micro-kernel output matches the reference implementation
bool is_output_correct(
    size_t num_rows, size_t num_cols, const float16_t tolerance, const float16_t* ref, const float16_t* act) {
    bool is_valid = true;

    for (size_t i = 0; i < num_rows * num_cols; ++i) {
        if (std::fabs(ref[i] - act[i]) > tolerance) {
            const size_t x = i % num_cols;
            const size_t y = i / num_cols;

            std::cout << std::setprecision(5) << std::fixed << "ERROR![" << y << "][" << x << "]: ref=" << ref[i]
                      << " vs. act=" << act[i] << "\n";

            is_valid = false;
        }
    }
    return is_valid;
}
}  // namespace

// ─── QMX MOPA ukernel variant table ─────────────────────────────────────────

struct kai_matmul_ukernel_f16_f16p_f16p_qmx {
    kai_matmul_clamp_f16_f16p_f16p_ukernel ukernel;
    std::string name = {};
};

static kai_matmul_ukernel_f16_f16p_f16p_qmx qmx_ukernel_variants[] = {
    {
        {
            kai_get_m_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_n_step_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_nr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_kr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_sr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_lhs_packed_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_rhs_packed_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_dst_offset_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_dst_size_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa,
            kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa,
        },
        "matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_qmx_mopa",
    },
};

static const size_t num_qmx_ukernel_variants =
    sizeof(qmx_ukernel_variants) / sizeof(qmx_ukernel_variants[0]);

// ─────────────────────────────────────────────────────────────────────────────

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
    int ret = 0;

    const size_t num_threads = parse_thread_count(argc, argv);
    std::cout << "Using " << num_threads << " thread(s) for computations.\n";

    // 1x1 Convolution operator in NHWC format.
    const size_t nhwc_n = 2;
    const size_t nhwc_h = 2;
    const size_t nhwc_w = 4;
    const size_t nhwc_c_in = 4;    // Input channels
    const size_t nhwc_c_out = 24;  // Output channels

    // Map NHWC of operator to GEMM terminology
    const size_t M = nhwc_h * nhwc_w * nhwc_n;  // Rows of LHS and DST matrices
    const size_t N = nhwc_c_out;                // Columns of RHS and DST matrices
    const size_t K = nhwc_c_in;                 // Columns of LHS, rows of RHS matrices

    const size_t lhs_size = M * K;
    const size_t rhs_size = N * K;
    const size_t bias_size = N;
    const size_t dst_size = M * N;

    // Allocate the memory
    float16_t* lhs = new float16_t[lhs_size];
    float16_t* rhs = new float16_t[rhs_size];
    float16_t* bias = new float16_t[bias_size];

    fill_matrix(M, K, lhs, 0.1);
    fill_matrix(K, N, rhs, 0.1);
    fill_matrix(1, N, bias, 10);

#ifdef KAI_DEBUG
    print_matrix(M, K, "lhs", lhs);
    print_matrix(K, N, "rhs", rhs);
    print_matrix(1, N, "bias", bias);
#endif  // KAI_DEBUG

    //----------- REFERENCE IMPLEMENTATION
    //------------------------------------
    //------------------------------------
    float16_t* dst_ref = new float16_t[dst_size];

    run_matmul_ref(
        M, N, K,           // Dimensions
        lhs,               // LHS buffer
        rhs,               // RHS buffer
        bias,              // Bias buffer
        dst_ref,           // DST
        -FLT_MAX, FLT_MAX  // Min and max for the clamp operation
    );
    //----------- END REFERENCE IMPLEMENTATION
    //------------------------------------
    //------------------------------------

    //----------- MICRO-KERNELS TESTS
    //------------------------------------
    //------------------------------------
    const size_t nr = ukernel.get_nr();
    const size_t kr = ukernel.get_kr();
    const size_t sr = ukernel.get_sr();

    // In a single row, we pack nr bias values followed by K rows of nr RHS values
    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(N, K);
    const size_t rhs_packed_cols = nr + K * nr;
    const size_t rhs_packed_rows = rhs_packed_size / (rhs_packed_cols * sizeof(float16_t));

    uint8_t* rhs_packed = new uint8_t[rhs_packed_size];

    const size_t lhs_stride = K * sizeof(float16_t);
    const size_t rhs_stride = N * sizeof(float16_t);
    const size_t dst_stride_row = N * sizeof(float16_t);
    const size_t dst_stride_col = sizeof(float16_t);

    // Packing only needs to be performed once if the contents of the bias and RHS matrices are expected to be constant.
    kai_run_rhs_pack_kxn_f16p16x1biasf16_f16_f16_neon(
        1, N, K, nr, kr, sr,  // Packing arguments
        rhs_stride,           // RHS stride
        rhs,                  // RHS
        bias,                 // Bias
        NULL,                 // Scale
        rhs_packed,           // RHS packed
        0, NULL);

    // The RHS and Bias buffers can be freed after packing, however we reuse them for the reference test below

#ifdef KAI_DEBUG
    print_matrix(rhs_packed_rows, rhs_packed_cols, "rhs_packed", rhs_packed);
#endif  // KAI_DEBUG

    float16_t* dst = new float16_t[dst_size];

    // Framework scheduling params

    // Example alternative values to try. ukernel.get_m_step() * 2 or M;
    const size_t m_step = ukernel.get_m_step();  // Scheduling along M

    // Example alternative values to try.  n_step = N;
    const size_t n_step = ukernel.get_n_step();  // Scheduling along N

    for (size_t i_m_step = 0; i_m_step < M; i_m_step += m_step) {
        for (size_t i_n_step = 0; i_n_step < N; i_n_step += n_step) {
            // Support functions return offset in bytes
            const uint8_t* lhs_ptr =
                (const uint8_t*)lhs + (ukernel.get_lhs_packed_offset(i_m_step, K * sizeof(uint16_t)));
            const uint8_t* rhs_ptr = (const uint8_t*)rhs_packed + (ukernel.get_rhs_packed_offset(i_n_step, K));
            uint8_t* dst_ptr = (uint8_t*)dst + (ukernel.get_dst_offset(i_m_step, i_n_step, N * sizeof(uint16_t)));
#ifdef KAI_DEBUG
            printf("Processing a %zux%zu ouptut block starting at (%zu, %zu)\n", m_step, n_step, i_m_step, i_n_step);
#endif
            const size_t actual_m = std::min(M - i_m_step, m_step);
            const size_t actual_n = std::min(N - i_n_step, n_step);

            ukernel.run_matmul(
                actual_m, actual_n, K,  // Dimensions
                lhs_ptr,                // LHS
                lhs_stride,             // LHS stride
                rhs_ptr,                // RHS packed
                dst_ptr,                // DST
                dst_stride_row,         // DST stride (row)
                dst_stride_col,         // DST stride (col)
                -FLT_MAX, FLT_MAX       // Min and max for the clamp operation
            );
        }
    }

#ifdef KAI_DEBUG
    print_matrix(M, N, "dst", dst);
#endif  // KAI_DEBUG

    const bool is_valid = is_output_correct(M, N, 0.0001, dst_ref, dst);

    std::cout << "TEST[matmul_clamp_f16_f16_f16p]\n";
    std::cout << "- ukernel: matmul_clamp_f16_f16_f16p16x1biasf16_6x16x8_neon_mla\n";
    if (is_valid) {
        std::cout << "- Status: PASSED\n";
    } else {
        std::cout << "- Status: FAILED\n";
        ret = 1;
    }

    //----------- END MICRO-KERNELS TESTS
    //------------------------------------
    //------------------------------------

    delete[] lhs;
    delete[] rhs;
    delete[] bias;
    delete[] rhs_packed;
    delete[] dst;
    delete[] dst_ref;

    // ── QMX MOPA kernel test (multi-threaded, 16 iterations) ─────────────────
    // Allocate fresh input matrices for the QMX MOPA test (M=512, N=512, K=512)
    {
        const size_t M_qmx = 512;
        const size_t N_qmx = 512;
        const size_t K_qmx = 512;

        std::cout << "\nTEST QMX MOPA [M=" << M_qmx << " N=" << N_qmx << " K=" << K_qmx << "]\n";

        float16_t* lhs_qmx  = new float16_t[M_qmx * K_qmx];
        float16_t* rhs_qmx  = new float16_t[K_qmx * N_qmx];
        float16_t* bias_qmx = new float16_t[N_qmx];

        fill_matrix(M_qmx, K_qmx, lhs_qmx,  float16_t(0.001f));
        fill_matrix(K_qmx, N_qmx, rhs_qmx,  float16_t(0.001f));
        fill_matrix(1,     N_qmx, bias_qmx, float16_t(0.1f));

        // Reference: accumulate in FP32 to avoid FP16 precision loss at large K
        float16_t* dst_ref_qmx = new float16_t[M_qmx * N_qmx];
        for (size_t row_idx = 0; row_idx < M_qmx; ++row_idx) {
            for (size_t col_idx = 0; col_idx < N_qmx; ++col_idx) {
                float acc = static_cast<float>(bias_qmx[col_idx]);
                for (size_t k_idx = 0; k_idx < K_qmx; ++k_idx) {
                    acc += static_cast<float>(lhs_qmx[row_idx * K_qmx + k_idx]) *
                           static_cast<float>(rhs_qmx[col_idx + N_qmx * k_idx]);
                }
                acc = std::max(acc, -FLT_MAX);
                acc = std::min(acc, FLT_MAX);
                dst_ref_qmx[row_idx * N_qmx + col_idx] = static_cast<float16_t>(acc);
            }
        }

    for (size_t idx_variant = 0; idx_variant < num_qmx_ukernel_variants; ++idx_variant) {
        std::cout << "Testing " << qmx_ukernel_variants[idx_variant].name << "\n";

        const size_t mr = qmx_ukernel_variants[idx_variant].ukernel.get_mr();
        const size_t nr = qmx_ukernel_variants[idx_variant].ukernel.get_nr();
        const size_t kr = qmx_ukernel_variants[idx_variant].ukernel.get_kr();
        const size_t sr = qmx_ukernel_variants[idx_variant].ukernel.get_sr();

        const size_t lhs_packed_size =
            kai_get_lhs_packed_size_lhs_pack_x16p2vlx2_x16_sme(M_qmx, K_qmx, mr, kr, sr);
        const size_t rhs_packed_size =
            kai_get_rhs_packed_size_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(N_qmx, K_qmx);
        const size_t dst_size_bytes =
            qmx_ukernel_variants[idx_variant].ukernel.get_dst_size(M_qmx, N_qmx);

        uint8_t*   lhs_packed_qmx = new uint8_t[lhs_packed_size];
        uint8_t*   rhs_packed_qmx = new uint8_t[rhs_packed_size];
        float16_t* dst_qmx        = new float16_t[dst_size_bytes / sizeof(float16_t)];

        // Pack RHS + bias once (constant weights, done before threading)
        const size_t rhs_stride_qmx = N_qmx * sizeof(float16_t);
        kai_run_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme(
            1, N_qmx, K_qmx, nr, kr, sr,
            rhs_stride_qmx, rhs_qmx, bias_qmx,
            /*scale=*/NULL, rhs_packed_qmx,
            /*extra_bytes=*/0, /*params=*/NULL);

        // ── Phase 1: pack LHS once (not timed) ───────────────────────────────
        auto lhs_pack_worker = [&](int thread_index) {
            const size_t m_step =
                qmx_ukernel_variants[idx_variant].ukernel.get_m_step();
            const size_t num_m_per_thread =
                kai_roundup(M_qmx, m_step * num_threads) / num_threads;
            const size_t m_start = static_cast<size_t>(thread_index) * num_m_per_thread;
            if (m_start >= M_qmx) return;
            const size_t m_to_process = std::min(num_m_per_thread, M_qmx - m_start);

            const size_t lhs_src_offset =
                kai_get_lhs_offset_lhs_pack_x16p2vlx2_x16_sme(
                    m_start, K_qmx * sizeof(float16_t));
            const void* lhs_src_ptr =
                reinterpret_cast<const uint8_t*>(lhs_qmx) + lhs_src_offset;

            const size_t lhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, K_qmx);
            void* lhs_packed_ptr = lhs_packed_qmx + lhs_packed_offset;

            kai_run_lhs_pack_x16p2vlx2_x16_sme(
                m_to_process, K_qmx, mr, kr, sr,
                /*m_idx_start=*/0,
                lhs_src_ptr, K_qmx * sizeof(float16_t),
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
                kai_roundup(M_qmx, m_step * num_threads) / num_threads;
            const size_t m_start = static_cast<size_t>(thread_index) * num_m_per_thread;
            if (m_start >= M_qmx) return;
            const size_t m_to_process = std::min(num_m_per_thread, M_qmx - m_start);

            const size_t dst_stride_row = N_qmx * sizeof(float16_t);

            const size_t lhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, K_qmx);
            const void* lhs_packed_ptr = lhs_packed_qmx + lhs_packed_offset;

            const size_t rhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, K_qmx);
            const void* rhs_packed_ptr = rhs_packed_qmx + rhs_packed_offset;

            const size_t dst_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_dst_offset(
                    m_start, 0, dst_stride_row);
            void* dst_ptr = reinterpret_cast<uint8_t*>(dst_qmx) + dst_offset;

            // Each thread runs run_matmul num_iterations times.
            // join() in the main thread blocks until this loop completes,
            // so time_e - time_s covers all num_iterations calls.
            for (int iter = 0; iter < num_iterations; ++iter) {
                qmx_ukernel_variants[idx_variant].ukernel.run_matmul(
                    m_to_process, N_qmx, K_qmx,
                    lhs_packed_ptr, rhs_packed_ptr,
                    dst_ptr,
                    dst_stride_row, sizeof(float16_t),
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

        // 0.5% relative tolerance (correctness checked on the last iteration's output)
        bool is_valid_qmx = true;
        for (size_t i = 0; i < M_qmx * N_qmx; ++i) {
            const float ref_f = static_cast<float>(dst_ref_qmx[i]);
            const float act_f = static_cast<float>(dst_qmx[i]);
            const float abs_diff = std::fabs(ref_f - act_f);
            const float threshold = 0.005f * std::max(std::fabs(ref_f), 1.0f);
            if (abs_diff > threshold) {
                const size_t x = i % N_qmx;
                const size_t y = i / N_qmx;
                std::cout << std::setprecision(5) << std::fixed
                          << "ERROR![" << y << "][" << x << "]: ref=" << ref_f
                          << " vs. act=" << act_f << "\n";
                is_valid_qmx = false;
            }
        }

        std::cout << "TEST[matmul_clamp_f16_f16p_f16p]\n";
        std::cout << "- ukernel: " << qmx_ukernel_variants[idx_variant].name << "\n";
        if (is_valid_qmx) {
            const double gflops = compute_gflops(M_qmx, N_qmx, K_qmx, avg_us);
            std::cout << "- Status: PASSED\n";
            std::cout << "- Iterations: " << num_iterations << "\n";
            std::cout << "- Total Performance time: " << total_us << " us\n";
            std::cout << "- Avg Performance time per iteration: " << avg_us << " us\n";
            std::cout << std::fixed << std::setprecision(2) << "- GFLOPS: " << gflops << "\n";
        } else {
            std::cout << "- Status: FAILED\n";
            ret = 1;
        }
        std::cout << "------------\n";

        delete[] lhs_packed_qmx;
        delete[] rhs_packed_qmx;
        delete[] dst_qmx;
    }  // end QMX variant loop

        delete[] lhs_qmx;
        delete[] rhs_qmx;
        delete[] bias_qmx;
        delete[] dst_ref_qmx;
    }  // end QMX input block

    return ret;
}
#endif  // Architectural features check.
