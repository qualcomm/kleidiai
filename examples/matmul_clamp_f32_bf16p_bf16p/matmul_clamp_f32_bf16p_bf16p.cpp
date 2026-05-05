//
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Example usage for matrix multiplication of two half-precision brain floating-point (BF16) matrices
// and the accumulation of the result into an FP32 destination matrix.
//
// The activations and the weights, stored in the LHS and RHS matrices respectively, are both non-transposed matrices.
// The matrix multiplication computation is performed using BF16 matrix multiply (BFMMLA)
// vector instructions present in the FEAT_BF16 Arm® architecture feature.
//
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_BF16_SCALAR_ARITHMETIC) || \
    !defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#error This file must be compiled for AArch64, FEAT_BF16.
#else
#include <arm_neon.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

// Include micro-kernel variants
#include "kai/kai_common.h"
#include "kai_lhs_quant_pack_bf16p1x4_f32_neon.h"
#include "kai_lhs_quant_pack_bf16p8x4_f32_neon.h"
#include "kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot.h"
#include "kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"
#include "kai_matmul_clamp_f32_bf16p_bf16p_interface.h"
#include "kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.h"

// QMX MOPA kernel – packed LHS (SME) + packed RHS+bias (SME)
#include "kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa.h"
#include "kai_lhs_pack_bf16p2vlx2_f32_sme.h"
#include "kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.h"

inline static float bf16_to_float(const uint16_t* v) {
    const uint16_t uint_rep = *v;
    return kai_cast_f32_bf16(uint_rep);
}

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

typedef void (*kai_lhs_quant_pack_bf16pmxk_f32_run_func_t)(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr, size_t m_idx_start, const void* lhs, size_t lhs_stride,
    void* lhs_packed);

typedef size_t (*kai_lhs_quant_pack_bf16pmxk_f32_get_lhs_packed_size_func_t)(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr);

struct kai_matmul_clamp_f32_bf16p_bf16p {
    kai_matmul_clamp_f32_bf16p_bf16p_ukernel matmul_ukernel;
    kai_lhs_quant_pack_bf16pmxk_f32_run_func_t lhs_pack_ukernel;
    kai_lhs_quant_pack_bf16pmxk_f32_get_lhs_packed_size_func_t lhs_pack_get_lhs_packed_size;
    std::string name = {};
};

/// Micro-kernel interface
const kai_matmul_clamp_f32_bf16p_bf16p ukernel_variants[] = {
    {{kai_get_m_step_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
      kai_get_n_step_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
      kai_get_mr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
      kai_get_nr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
      kai_get_kr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
      kai_get_sr_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
      kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
      kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
      kai_get_dst_offset_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
      kai_get_dst_size_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot,
      kai_run_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot},
     kai_run_lhs_quant_pack_bf16p1x4_f32_neon,
     kai_get_lhs_packed_size_lhs_quant_pack_bf16p1x4_f32_neon,
     "matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot"},
    {{kai_get_m_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
      kai_get_n_step_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
      kai_get_mr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
      kai_get_nr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
      kai_get_kr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
      kai_get_sr_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
      kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
      kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
      kai_get_dst_offset_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
      kai_get_dst_size_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla,
      kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla},
     kai_run_lhs_quant_pack_bf16p8x4_f32_neon,
     kai_get_lhs_packed_size_lhs_quant_pack_bf16p8x4_f32_neon,
     "matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla"}};

// Number of micro-kernel variants stored in the array
constexpr size_t num_ukernel_variants = sizeof(ukernel_variants) / sizeof(ukernel_variants[0]);

/// @brief Truncate the 32-bit floating point number's least significant 16 mantissa bits
/// @param x floating-point number
/// @return truncated floating-point number
inline static float truncate(float x) {
    uint32_t uval = (*reinterpret_cast<uint32_t*>(&x) & 0xffff0000);
    return *reinterpret_cast<float*>(&uval);
}

/// Reference implementation of matrix multiplication
static void run_matmul_ref(
    size_t m, size_t n, size_t k, const float* lhs, const float* rhs, const float* bias, float* dst, float scalar_min,
    float scalar_max) {
    for (size_t row_idx = 0; row_idx < m; ++row_idx) {
        for (size_t col_idx = 0; col_idx < n; ++col_idx) {
            float acc = bias[col_idx];

            for (size_t k_idx = 0; k_idx < k; ++k_idx) {
                float lhs_val = truncate(lhs[row_idx * k + k_idx]);
                float rhs_val = truncate(rhs[col_idx + n * k_idx]);

                acc += lhs_val * rhs_val;
            }

            dst[row_idx * n + col_idx] = std::clamp(acc, scalar_min, scalar_max);
        }
    }
}

/// Fills the matrix with incremental values
void fill_matrix(size_t num_rows, size_t num_cols, float* dst, const float weight) {
    for (size_t i = 0; i < num_rows * num_cols; i++) {
        dst[i] = float((i + 1) * weight);
    }
}

/// Print the matrix
void print_matrix(size_t num_rows, size_t num_cols, const char* name, const float* src) {
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

void print_matrix(size_t num_rows, size_t num_cols, const char* name, const uint16_t* src) {
    std::cout << name << " = [\n";
    for (size_t y = 0; y < num_rows; ++y) {
        std::cout << "  [";
        for (size_t x = 0; x < num_cols; ++x) {
            std::cout << std::setprecision(2) << std::fixed << bf16_to_float(&src[y * num_cols + x]) << ", ";
        }
        std::cout << ("],\n");
    }
    std::cout << ("]\n\n");
}

void print_mixed_prec_matrix(
    size_t num_rows, size_t num_cols, const char* name, const uint8_t* src, int nr, int stride) {
    std::cout << name << " = [\n";
    for (size_t y = 0; y < num_rows; ++y) {
        std::cout << "  [";
        const uint8_t* src_row = src + stride * y;
        for (size_t x = 0; x < num_cols; ++x) {
            if (x >= nr) {
                // print bfloat
                const uint16_t* src_elm =
                    reinterpret_cast<const uint16_t*>(src_row + nr * sizeof(float) + (x - nr) * sizeof(uint16_t));
                std::cout << std::setprecision(2) << std::fixed << bf16_to_float(src_elm) << ", ";
            } else {
                // print float
                const float* src_elm = reinterpret_cast<const float*>(src_row + x * sizeof(float));
                std::cout << std::setprecision(2) << std::fixed << *src_elm << ", ";
            }
        }
        std::cout << ("],\n");
    }
    std::cout << ("]\n\n");
}

void print_bf_matrix(size_t num_rows, size_t num_cols, const char* name, const float* src) {
    std::cout << name << " = [\n";
    for (size_t y = 0; y < num_rows; ++y) {
        std::cout << "  [";
        for (size_t x = 0; x < num_cols; ++x) {
            std::cout << std::setprecision(2) << std::fixed << truncate(src[y * num_cols + x]) << ", ";
        }
        std::cout << ("],\n");
    }
    std::cout << ("]\n\n");
}

/// Verify the micro-kernel output matches the reference implementation
bool is_output_correct(
    size_t num_rows, size_t num_cols, const float rel_tolerance, const float* ref, const float* act) {
    bool is_valid = true;

    for (size_t i = 0; i < num_rows * num_cols; ++i) {
        if (std::fabs(ref[i] - act[i]) / (act[i] + 1e-10) > rel_tolerance) {
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

struct kai_matmul_ukernel_f32_bf16p_bf16p_qmx {
    kai_matmul_clamp_f32_bf16p_bf16p_ukernel ukernel;
    std::string name = {};
};

static kai_matmul_ukernel_f32_bf16p_bf16p_qmx qmx_ukernel_variants[] = {
    {
        {
            kai_get_m_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_n_step_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_mr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_nr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_kr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_sr_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_lhs_packed_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_rhs_packed_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_dst_offset_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa,
            kai_get_dst_size_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa,
            kai_run_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa,
        },
        "matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa",
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

    // Matrix dimensions
    const size_t M = 512;  // LHS rows / output rows
    const size_t N = 512;  // RHS columns / output columns
    const size_t K = 512;  // Common dimension (must be even for QMX MOPA kr=2)

    std::cout << "Matrix dimensions: M=" << M << " N=" << N << " K=" << K << "\n\n";

    for (int variant_idx = 0; variant_idx < num_ukernel_variants; ++variant_idx) {
        const size_t lhs_size = M * K;
        const size_t rhs_size = N * K;
        const size_t bias_size = N;
        const size_t dst_size = M * N;

        const auto ukernel = ukernel_variants[variant_idx].matmul_ukernel;
        const auto lhs_pack_ukernel = ukernel_variants[variant_idx].lhs_pack_ukernel;
        const auto get_lhs_packed_size = ukernel_variants[variant_idx].lhs_pack_get_lhs_packed_size;

        // Allocate the memory
        float* lhs = new float[lhs_size];
        float* rhs = new float[rhs_size];
        float* bias = new float[bias_size];

        fill_matrix(M, K, lhs, 0.4);
        fill_matrix(K, N, rhs, 0.3);
        fill_matrix(1, N, bias, 0.2);

#ifdef KAI_DEBUG
        print_matrix(M, K, "lhs", lhs);
        print_matrix(K, N, "rhs", rhs);
        print_matrix(1, N, "bias", bias);

        // Print bf16 converted values
        print_bf_matrix(M, K, "lhs_bf", lhs);
        print_bf_matrix(K, N, "rhs_bf", rhs);
#endif  // KAI_DEBUG

        //----------- REFERENCE IMPLEMENTATION
        //------------------------------------
        //------------------------------------
        float* dst_ref = new float[dst_size];

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
        const size_t mr = ukernel.get_mr();
        const size_t nr = ukernel.get_nr();
        const size_t kr = ukernel.get_kr();
        const size_t sr = ukernel.get_sr();

        // In a single row, we pack nr bias values followed by K rows of nr RHS values
        const size_t rhs_packed_size =
            kai_get_rhs_packed_size_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(N, K, nr, kr);
        uint8_t* rhs_packed = new uint8_t[rhs_packed_size];

        const size_t lhs_stride = K * sizeof(float);
        const size_t rhs_stride = N * sizeof(float);
        const size_t dst_stride_row = N * sizeof(float);
        const size_t dst_stride_col = sizeof(float);

        // Packing only needs to be performed once if the contents of the bias and RHS matrices are expected to be
        // constant.
        kai_run_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon(
            1, N, K, nr, kr, sr,  // Packing arguments
            rhs_stride,           // RHS stride
            rhs,                  // RHS
            bias,                 // Bias
            NULL,                 // Scale
            rhs_packed,           // RHS packed
            0, NULL);

        // The RHS and Bias buffers can be freed after packing, however we reuse them for the reference test below

#ifdef KAI_DEBUG
        const size_t rhs_packed_cols = nr + kai_roundup(K, kr) * nr;

        // Each col has nr floats and then K*nr bfloats
        int rhs_packed_stride = nr * sizeof(float) + kai_roundup(K, kr) * nr * sizeof(uint16_t);
        const size_t rhs_packed_rows = rhs_packed_size / rhs_packed_stride;

        print_mixed_prec_matrix(rhs_packed_rows, rhs_packed_cols, "rhs_packed", rhs_packed, nr, rhs_packed_stride);
#endif  // KAI_DEBUG

        float* dst = new float[dst_size];

        const auto timer_matmul_start = std::chrono::high_resolution_clock::now();

        // This can be anything for GEMM kernels. It does not have to be equal to m_step() returned
        // from the kernel. But, for GEMV, it must be m_step (which will be equal to 1).
        const size_t m_step = ukernel.get_m_step();
        for (size_t m_idx = 0; m_idx < M; m_idx += m_step) {
            const size_t height = KAI_MIN(m_step, M - m_idx);

            size_t lhs_packed_size = get_lhs_packed_size(height, K, mr, kr, sr);

            uint8_t* lhs_packed = new uint8_t[lhs_packed_size];
            memset(lhs_packed, 0, lhs_packed_size);

            lhs_pack_ukernel(
                height, K, mr, kr, sr, 0 /* m_idx_start */, reinterpret_cast<uint8_t*>(lhs) + m_idx * lhs_stride,
                lhs_stride, lhs_packed);

#ifdef KAI_DEBUG
            int num_lhs_rows = (height + mr - 1) / mr;
            int num_lhs_cols = mr * kai_roundup(K, kr);

            print_matrix(num_lhs_rows, num_lhs_cols, "lhs_packed", reinterpret_cast<uint16_t*>(lhs_packed));
#endif  // KAI_DEBUG

            ukernel.run_matmul(
                height, N, K,                                              // Dimensions
                lhs_packed,                                                // LHS packed
                rhs_packed,                                                // RHS packed
                reinterpret_cast<uint8_t*>(dst) + m_idx * dst_stride_row,  // DST
                dst_stride_row,                                            // DST stride (row)
                dst_stride_col,                                            // DST stride (col)
                -FLT_MAX, FLT_MAX                                          // Min and max for the clamp operation
            );

            delete[] lhs_packed;
        }

        const auto timer_matmul_end = std::chrono::high_resolution_clock::now();
        const auto time_matmul =
            std::chrono::duration_cast<std::chrono::nanoseconds>(timer_matmul_end - timer_matmul_start);

#ifdef KAI_DEBUG
        print_matrix(M, N, "dst", dst);
        print_matrix(M, N, "ref", dst_ref);
#endif  // KAI_DEBUG

        constexpr float rel_tolerance = 0.02;  // This value was chosen by experimentation
        const bool is_valid = is_output_correct(M, N, rel_tolerance, dst_ref, dst);

        std::cout << "TEST[matmul_clamp_f32_bf16p_bf16p]\n";
        std::cout << "- ukernel: " << ukernel_variants[variant_idx].name << std::endl;
        if (is_valid) {
            std::cout << "- Status: PASSED\n";
            std::cout << "- Performance: " << time_matmul.count() << "ns\n";
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
    }

    // ── TEST[2..N]: QMX MOPA kernel (packed LHS + packed RHS, multi-threaded) ─
    // Allocate fresh input matrices for the QMX MOPA test (the NEON loop freed its copies)
    {
        const size_t lhs_size_qmx  = M * K;
        const size_t rhs_size_qmx  = K * N;
        const size_t bias_size_qmx = N;
        const size_t dst_size_qmx  = M * N;

        float* lhs_qmx_in  = new float[lhs_size_qmx];
        float* rhs_qmx_in  = new float[rhs_size_qmx];
        float* bias_qmx_in = new float[bias_size_qmx];

        fill_matrix(M, K, lhs_qmx_in,  0.4f);
        fill_matrix(K, N, rhs_qmx_in,  0.3f);
        fill_matrix(1, N, bias_qmx_in, 0.2f);

        // Compute reference output for QMX MOPA correctness check
        float* dst_ref_qmx = new float[dst_size_qmx];
        run_matmul_ref(M, N, K, lhs_qmx_in, rhs_qmx_in, bias_qmx_in, dst_ref_qmx, -FLT_MAX, FLT_MAX);

    for (size_t idx_variant = 0; idx_variant < num_qmx_ukernel_variants; ++idx_variant) {
        const size_t test_idx = idx_variant + 2;
        std::cout << "Testing " << qmx_ukernel_variants[idx_variant].name << "\n";

        const size_t mr = qmx_ukernel_variants[idx_variant].ukernel.get_mr();
        const size_t nr = qmx_ukernel_variants[idx_variant].ukernel.get_nr();
        const size_t kr = qmx_ukernel_variants[idx_variant].ukernel.get_kr();
        const size_t sr = qmx_ukernel_variants[idx_variant].ukernel.get_sr();

        const size_t lhs_packed_size =
            kai_get_lhs_packed_size_lhs_pack_bf16p2vlx2_f32_sme(M, K, mr, kr, sr);
        const size_t rhs_packed_size =
            kai_get_rhs_packed_size_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(N, K);
        const size_t dst_size_bytes =
            qmx_ukernel_variants[idx_variant].ukernel.get_dst_size(M, N);

        uint8_t* lhs_packed_qmx = new uint8_t[lhs_packed_size];
        uint8_t* rhs_packed_qmx = new uint8_t[rhs_packed_size];
        float*   dst_qmx        = new float[dst_size_bytes / sizeof(float)];

        // Pack RHS + bias once (constant weights, done before threading)
        const size_t rhs_stride_qmx = N * sizeof(float);
        kai_run_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme(
            1, N, K, nr, kr, sr,
            rhs_stride_qmx, rhs_qmx_in, bias_qmx_in,
            /*scale=*/NULL, rhs_packed_qmx,
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
                kai_get_lhs_offset_lhs_pack_bf16p2vlx2_f32_sme(
                    m_start, K * sizeof(float));
            const void* lhs_src_ptr =
                reinterpret_cast<const uint8_t*>(lhs_qmx_in) + lhs_src_offset;

            const size_t lhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_lhs_packed_offset(
                    m_start, K);
            void* lhs_packed_ptr = lhs_packed_qmx + lhs_packed_offset;

            kai_run_lhs_pack_bf16p2vlx2_f32_sme(
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
            const void* lhs_packed_ptr = lhs_packed_qmx + lhs_packed_offset;

            const size_t rhs_packed_offset =
                qmx_ukernel_variants[idx_variant].ukernel.get_rhs_packed_offset(0, K);
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
                    m_to_process, N, K,
                    lhs_packed_ptr, rhs_packed_ptr,
                    dst_ptr,
                    dst_stride_row, sizeof(float),
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

        // 2% relative tolerance (correctness checked on the last iteration's output)
        constexpr float rel_tolerance_qmx = 0.02f;
        const bool is_valid_qmx =
            is_output_correct(M, N, rel_tolerance_qmx, dst_ref_qmx, dst_qmx);

        printf("TEST[%zu] = %s\n", test_idx, is_valid_qmx ? "PASSED" : "FAILED");
        std::cout << "- ukernel: " << qmx_ukernel_variants[idx_variant].name << "\n";
        std::cout << "- Iterations: " << num_iterations << "\n";
        std::cout << "- Total time: " << total_us << " us\n";
        const double gflops = compute_gflops(M, N, K, avg_us);
        std::cout << "- Avg time per iteration: " << avg_us << " us\n";
        std::cout << std::fixed << std::setprecision(2) << "- GFLOPS: " << gflops << "\n\n";
        if (!is_valid_qmx) ret = 1;

        delete[] lhs_packed_qmx;
        delete[] rhs_packed_qmx;
        delete[] dst_qmx;
    }  // end QMX variant loop

        delete[] lhs_qmx_in;
        delete[] rhs_qmx_in;
        delete[] bias_qmx_in;
        delete[] dst_ref_qmx;
    }  // end QMX input block

    return ret;
}

#endif  // Architectural features check.
