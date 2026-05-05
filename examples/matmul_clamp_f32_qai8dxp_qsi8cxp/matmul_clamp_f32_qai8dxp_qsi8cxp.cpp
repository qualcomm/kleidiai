//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// + Changes from Qualcomm Technologies, Inc. are provided under the following license:
// + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// + SPDX-License-Identifier: BSD-3-Clause-Clear
//

// Example usage for matrix multiplication with dynamically quantized 8-bit asymmetric per-row LHS (qai8dxp)
// and 8-bit symmetric per-channel RHS (qsi8cxp), producing FP32 output.
//
// Tests one QMX MOPA micro-kernel variant:
//   TEST[0]: kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa
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
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "kai/kai_common.h"
#include "kai_lhs_quant_pack_qai8dxp_f32.h"
#include "kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa.h"
#include "kai_matmul_clamp_f32_qai8dxp_qsi8cxp_interface.h"
#include "kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"

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

struct kai_matmul_ukernel_f32_qa8dxp_qs8cxp_qmx {
    kai_matmul_clamp_f32_qai8dxp_qsi8cxp_ukernel ukernel;
    std::string name = {};
};

static kai_matmul_ukernel_f32_qa8dxp_qs8cxp_qmx qmx_ukernel_variants[] = {
    {{kai_get_m_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa,
      kai_get_n_step_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa,
      kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa,
      kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa,
      kai_get_kr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa,
      kai_get_sr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa,
      kai_get_lhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa,
      kai_get_rhs_packed_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa,
      kai_get_dst_offset_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa,
      kai_get_dst_size_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa,
      kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa},
     "matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_qmx_mopa"},
};

static const size_t num_qmx_ukernel_variants =
    sizeof(qmx_ukernel_variants) / sizeof(qmx_ukernel_variants[0]);

// ─── Helper functions ─────────────────────────────────────────────────────────

static void fill_uniform_random(size_t num_rows, size_t num_cols, float* dst, size_t seed) {
    std::srand(seed);
    for (size_t i = 0; i < num_rows * num_cols; i++) {
        dst[i] = (float)((double)std::rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

/// Quantize RHS to int8 per-channel (symmetric, NxK layout)
static void quant_qs8cx_f32(size_t n, size_t k, const float* rhs_f32,
                             int8_t* rhs_qs8cx, float* rhs_scales) {
    for (size_t n_idx = 0; n_idx < n; ++n_idx) {
        const float* src = rhs_f32 + n_idx * k;
        float max_abs = 0.0f;
        for (size_t k_idx = 0; k_idx < k; ++k_idx)
            max_abs = std::max(max_abs, std::fabs(src[k_idx]));
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        rhs_scales[n_idx] = scale;
        const float recip = 1.0f / scale;
        for (size_t k_idx = 0; k_idx < k; ++k_idx) {
            int32_t v = (int32_t)std::round(src[k_idx] * recip);
            v = std::max(v, -127); v = std::min(v, 127);
            rhs_qs8cx[n_idx * k + k_idx] = (int8_t)v;
        }
    }
}

/// Quantize LHS to int8 per-row (asymmetric)
static void ref_quant_qa8dx_f32(size_t m, size_t k, const float* lhs_f32, int8_t* lhs_qa8dx) {
    const size_t dst_stride = k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t);
    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
        const float* src = lhs_f32 + m_idx * k;
        float max0 = -FLT_MAX, min0 = FLT_MAX;
        for (size_t k_idx = 0; k_idx < k; ++k_idx) { max0 = std::max(src[k_idx], max0); min0 = std::min(src[k_idx], min0); }
        const float qmin = (float)INT8_MIN, qmax = (float)INT8_MAX;
        const float rmin0 = std::min(0.0f, min0), rmax0 = std::max(0.0f, max0);
        const float scale0 = rmin0 == rmax0 ? 1.f : (qmax - qmin) / (rmax0 - rmin0);
        const float recip_scale0 = scale0 ? 1.0f / scale0 : 0.0f;
        const float d_min = rmin0 * scale0, d_max = rmax0 * scale0;
        float zp = (qmin + d_min + qmax + d_max > 0) ? qmin - d_min : qmax - d_max;
        zp = std::max(zp, qmin); zp = std::min(zp, qmax);
        const int32_t nudged_zp = lrintf(zp);
        int8_t* dst = (int8_t*)lhs_qa8dx + m_idx * dst_stride;
        *((float*)dst) = recip_scale0; dst += sizeof(float);
        *((int32_t*)dst) = -nudged_zp; dst += sizeof(int32_t);
        for (size_t k_idx = 0; k_idx < k; ++k_idx) {
            int32_t v = (int32_t)std::round(src[k_idx] * scale0) + nudged_zp;
            v = std::max(v, INT8_MIN); v = std::min(v, INT8_MAX);
            dst[k_idx] = (int8_t)v;
        }
    }
}

/// Reference scalar matmul: LHS=qai8dx, RHS=qsi8cx (NxK), output=float
static void ref_matmul_f32_qa8dx_qs8cx(
    size_t m, size_t n, size_t k,
    const int8_t* lhs_qa8dx, const int8_t* rhs_qs8cx,
    const float* rhs_scales, float* dst_f32,
    float scalar_min, float scalar_max) {
    const size_t lhs_stride = k * sizeof(int8_t) + sizeof(float) + sizeof(int32_t);
    for (size_t m_idx = 0; m_idx < m; ++m_idx) {
        const int8_t* lhs_ptr = lhs_qa8dx + m_idx * lhs_stride;
        const float lhs_scale = *(const float*)lhs_ptr; lhs_ptr += sizeof(float);
        const int32_t lhs_offset = *(const int32_t*)lhs_ptr; lhs_ptr += sizeof(int32_t);
        for (size_t n_idx = 0; n_idx < n; ++n_idx) {
            int32_t iacc = 0;
            const int8_t* rhs_ptr = rhs_qs8cx + n_idx * k;
            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                iacc += (int32_t)lhs_ptr[k_idx] * (int32_t)rhs_ptr[k_idx];
                iacc += lhs_offset * (int32_t)rhs_ptr[k_idx];
            }
            float acc = (float)iacc * rhs_scales[n_idx] * lhs_scale;
            acc = std::max(acc, scalar_min); acc = std::min(acc, scalar_max);
            dst_f32[m_idx * n + n_idx] = acc;
        }
    }
}

/// Verify output with relative tolerance
static bool is_output_correct(size_t num_rows, size_t num_cols, float tolerance,
                               const float* ref, const float* act) {
    bool is_valid = true;
    for (size_t i = 0; i < num_rows * num_cols; ++i) {
        const float abs_diff = std::fabs(ref[i] - act[i]);
        const float threshold = tolerance * std::max(std::fabs(ref[i]), 1.0f);
        if (abs_diff > threshold) {
            const size_t x = i % num_cols, y = i / num_cols;
            std::cout << std::setprecision(5) << std::fixed
                      << "ERROR![" << y << "][" << x << "]: ref=" << ref[i]
                      << " vs. act=" << act[i] << "\n";
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

    // Allocate and fill input matrices
    float* lhs_f32  = new float[M * K];
    float* rhs_f32  = new float[N * K];
    int8_t* rhs_qs8cx = new int8_t[N * K];
    float*  rhs_scales = new float[N];

    fill_uniform_random(M, K, lhs_f32, seed_lhs);
    fill_uniform_random(N, K, rhs_f32, seed_rhs);
    quant_qs8cx_f32(N, K, rhs_f32, rhs_qs8cx, rhs_scales);
    delete[] rhs_f32;

    // Reference implementation
    const size_t lhs_ref_size = M * (K + sizeof(float) + sizeof(int32_t));
    int8_t* lhs_ref = new int8_t[lhs_ref_size];
    float*  dst_ref = new float[M * N];

    ref_quant_qa8dx_f32(M, K, lhs_f32, lhs_ref);
    ref_matmul_f32_qa8dx_qs8cx(M, N, K, lhs_ref, rhs_qs8cx, rhs_scales, dst_ref, -FLT_MAX, FLT_MAX);
    delete[] lhs_ref;

    // ── QMX MOPA kernel (packed LHS + packed RHS, multi-threaded) ─────────────
    for (size_t idx_variant = 0; idx_variant < num_qmx_ukernel_variants; ++idx_variant) {
        std::cout << "Testing " << qmx_ukernel_variants[idx_variant].name << "\n";

        const size_t mr = qmx_ukernel_variants[idx_variant].ukernel.get_mr();
        const size_t nr = qmx_ukernel_variants[idx_variant].ukernel.get_nr();
        const size_t kr = qmx_ukernel_variants[idx_variant].ukernel.get_kr();
        const size_t sr = qmx_ukernel_variants[idx_variant].ukernel.get_sr();

        const size_t lhs_packed_size =
            kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(M, K, mr, kr, sr);
        const size_t rhs_packed_size =
            kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(N, K, nr, kr, sr);
        const size_t dst_size_bytes =
            qmx_ukernel_variants[idx_variant].ukernel.get_dst_size(M, N);

        uint8_t* lhs_packed = new uint8_t[lhs_packed_size];
        uint8_t* rhs_packed = new uint8_t[rhs_packed_size];
        float*   dst        = new float[dst_size_bytes / sizeof(float)];

        // Pack RHS once (constant weights, done before threading)
        struct kai_rhs_pack_qsi8cx_params rhs_params;
        rhs_params.lhs_zero_point = 1;
        rhs_params.scale_multiplier = 1.0f;

        kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
            1, N, K, nr, kr, sr,
            rhs_qs8cx,
            /*bias=*/NULL,
            rhs_scales,
            rhs_packed,
            /*extra_bytes=*/0,
            &rhs_params);

        // ── Phase 1: pack LHS once (not timed) ───────────────────────────────
        auto lhs_pack_worker = [&](int thread_index) {
            const size_t m_step = qmx_ukernel_variants[idx_variant].ukernel.get_m_step();
            const size_t num_m_per_thread = kai_roundup(M, m_step * num_threads) / num_threads;
            const size_t m_start = static_cast<size_t>(thread_index) * num_m_per_thread;
            if (m_start >= M) return;
            const size_t m_to_process = std::min(num_m_per_thread, M - m_start);

            const float* src_ptr = lhs_f32 +
                kai_get_lhs_offset_lhs_quant_pack_qai8dxp_f32(m_start, K * sizeof(float)) / sizeof(float);
            const size_t lhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, K);
            void* lhs_packed_ptr = lhs_packed + lhs_packed_offset;

            kai_run_lhs_quant_pack_qai8dxp_f32(
                m_to_process, K, mr, kr, sr, 0,
                src_ptr, K * sizeof(float),
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

            const size_t dst_stride_row = N * sizeof(float);
            const size_t lhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(m_start, K);
            const void* lhs_packed_ptr = lhs_packed + lhs_packed_offset;
            const size_t rhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, K);
            const void* rhs_packed_ptr = rhs_packed + rhs_packed_offset;
            const size_t dst_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_dst_offset(m_start, 0, dst_stride_row);
            float* dst_ptr = dst + dst_offset / sizeof(float);

            for (int iter = 0; iter < num_iterations; ++iter) {
                qmx_ukernel_variants[idx_variant].ukernel.run_matmul(
                    m_to_process, N, K,
                    lhs_packed_ptr, rhs_packed_ptr,
                    dst_ptr, dst_stride_row, sizeof(float),
                    -FLT_MAX, FLT_MAX);
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

        // 0.5% relative tolerance
        const bool is_valid = is_output_correct(M, N, 0.005f, dst_ref, dst);

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
    delete[] lhs_f32;
    delete[] rhs_qs8cx;
    delete[] rhs_scales;
    delete[] dst_ref;

    return ret;
}

#endif  // Architectural features check.
