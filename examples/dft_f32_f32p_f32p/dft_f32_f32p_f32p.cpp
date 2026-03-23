//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

// Example: 2D DFT (float32) using DFT matrices and KleidiAI SME GEMMs,
// compared against Ne10 2-pass FFT for both accuracy and performance.

#if !defined(__aarch64__)
#error This file must be compiled for AArch64.
#elif !defined(__ARM_FEATURE_SME) && !defined(__ANDROID__)
#error This file must be compiled with FEAT_SME enabled.
#else

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

extern "C" {
#include "NE10.h"
#include "kai_lhs_pack_f32p2vlx1_f32_sme.h"
#include "kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.h"
#include "kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.h"
}

extern "C" void __arm_tpidr2_save(void) {}

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr int kReps = 5;

struct CompareStats {
    double max_err{};
    double rmse{};
};

struct CaseResult {
    size_t n{};
    size_t m{};
    double kai_ms{};
    double ne10_ms{};
    double kai_gflops{};
    double ne10_gflops{};
    CompareStats err{};
};

void fill_random(size_t n, float* dst, uint32_t seed) {
    for (size_t i = 0; i < n; ++i) {
        const uint32_t h = (static_cast<uint32_t>(i) + seed) * 1664525u + 1013904223u;
        dst[i] = static_cast<float>(h & 0xFFFFu) / 32768.0f - 1.0f;
    }
}

void generate_dft_matrix(size_t n, float* fr, float* fi) {
    const float two_pi_over_n = 2.0f * kPi / static_cast<float>(n);
    for (size_t k = 0; k < n; ++k) {
        for (size_t i = 0; i < n; ++i) {
            const float angle = two_pi_over_n * static_cast<float>(k) * static_cast<float>(i);
            fr[k * n + i] = std::cos(angle);
            fi[k * n + i] = -std::sin(angle);
        }
    }
}

void run_real_gemm_sme(size_t m, size_t n, size_t k, const float* lhs, const float* rhs, float* dst) {
    const size_t mr = kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
    const size_t nr = kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
    const size_t kr = kai_get_kr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
    const size_t sr = kai_get_sr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();

    const size_t lhs_stride = k * sizeof(float);
    const size_t rhs_stride = n * sizeof(float);
    const size_t dst_stride_row = n * sizeof(float);
    const size_t dst_stride_col = sizeof(float);

    const size_t lhs_packed_size = kai_get_lhs_packed_size_lhs_pack_f32p2vlx1_f32_sme(m, k, mr, kr, sr);
    std::vector<uint8_t> lhs_packed(lhs_packed_size);
    kai_run_lhs_pack_f32p2vlx1_f32_sme(m, k, mr, kr, sr, 0, lhs, lhs_stride, lhs_packed.data());

    std::vector<float> zero_bias(n, 0.0f);
    const size_t rhs_packed_size = kai_get_rhs_packed_size_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(n, k);
    std::vector<uint8_t> rhs_packed(rhs_packed_size);
    kai_run_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme(
        1, n, k, nr, kr, sr, rhs_stride, rhs, zero_bias.data(), nullptr, rhs_packed.data(), 0, nullptr);

    const size_t m_step = kai_get_m_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();
    const size_t n_step = kai_get_n_step_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa();

    for (size_t i_m = 0; i_m < m; i_m += m_step) {
        for (size_t i_n = 0; i_n < n; i_n += n_step) {
            const uint8_t* lhs_ptr =
                lhs_packed.data() +
                kai_get_lhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(i_m, k);
            const uint8_t* rhs_ptr =
                rhs_packed.data() +
                kai_get_rhs_packed_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(i_n, k);
            uint8_t* dst_ptr =
                reinterpret_cast<uint8_t*>(dst) +
                kai_get_dst_offset_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(i_m, i_n, dst_stride_row);

            kai_run_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa(
                std::min(m - i_m, m_step),
                std::min(n - i_n, n_step),
                k,
                lhs_ptr,
                rhs_ptr,
                dst_ptr,
                dst_stride_row,
                dst_stride_col,
                -FLT_MAX,
                FLT_MAX);
        }
    }
}

void run_dft_2d(size_t n, size_t m, const float* x, const float* f_nr, const float* f_ni, const float* f_mr,
                const float* f_mi, float* x_hat_r, float* x_hat_i) {
    const size_t n_half = n / 2 + 1;

    std::vector<float> temp_r(n * m, 0.0f);
    std::vector<float> temp_i(n * m, 0.0f);

    run_real_gemm_sme(n_half, m, n, f_nr, x, temp_r.data());
    run_real_gemm_sme(n_half, m, n, f_ni, x, temp_i.data());

    for (size_t k = 1; k + 1 < n_half; ++k) {
        const size_t k_conj = n - k;
        for (size_t col = 0; col < m; ++col) {
            temp_r[k_conj * m + col] = temp_r[k * m + col];
            temp_i[k_conj * m + col] = -temp_i[k * m + col];
        }
    }

    std::vector<float> t_a(n_half * m);
    std::vector<float> t_b(m * m);
    for (size_t i = 0; i < t_a.size(); ++i) {
        t_a[i] = temp_r[i] + temp_i[i];
    }
    for (size_t i = 0; i < t_b.size(); ++i) {
        t_b[i] = f_mr[i] + f_mi[i];
    }

    std::vector<float> p1(n_half * m);
    std::vector<float> p2(n_half * m);
    std::vector<float> p3(n_half * m);

    run_real_gemm_sme(n_half, m, m, temp_r.data(), f_mr, p1.data());
    run_real_gemm_sme(n_half, m, m, temp_i.data(), f_mi, p2.data());
    run_real_gemm_sme(n_half, m, m, t_a.data(), t_b.data(), p3.data());

    for (size_t k = 0; k < n_half; ++k) {
        for (size_t col = 0; col < m; ++col) {
            const size_t idx = k * m + col;
            x_hat_r[idx] = p1[idx] - p2[idx];
            x_hat_i[idx] = p3[idx] - p1[idx] - p2[idx];
        }
    }

    for (size_t k1 = 1; k1 + 1 < n_half; ++k1) {
        const size_t k1_conj = n - k1;
        for (size_t k2 = 0; k2 < m; ++k2) {
            const size_t k2_src = (m - k2) % m;
            x_hat_r[k1_conj * m + k2] = x_hat_r[k1 * m + k2_src];
            x_hat_i[k1_conj * m + k2] = -x_hat_i[k1 * m + k2_src];
        }
    }
}

void run_ne10_2d_r2c_fft(size_t n, size_t m, size_t m_half, const float* x, ne10_fft_cpx_float32_t* out,
                         ne10_fft_r2c_cfg_float32_t cfg_r2c, ne10_fft_cfg_float32_t cfg_c2c,
                         ne10_fft_cpx_float32_t* col_buf) {
    std::vector<ne10_fft_cpx_float32_t> row_buf(m);
    std::vector<float> row_in(m);
    for (size_t r = 0; r < n; ++r) {
        std::memcpy(row_in.data(), &x[r * m], m * sizeof(float));
        ne10_fft_r2c_1d_float32(row_buf.data(), row_in.data(), cfg_r2c);
        for (size_t c = 0; c < m_half; ++c) {
            out[r * m_half + c] = row_buf[c];
        }
    }

    std::vector<ne10_fft_cpx_float32_t> col_in(n);
    for (size_t c = 0; c < m_half; ++c) {
        for (size_t r = 0; r < n; ++r) {
            col_in[r] = out[r * m_half + c];
        }
        ne10_fft_c2c_1d_float32(col_buf, col_in.data(), cfg_c2c, 0);
        for (size_t r = 0; r < n; ++r) {
            out[r * m_half + c] = col_buf[r];
        }
    }
}

CompareStats compare_half_spectrum(size_t n, size_t m, const std::vector<float>& kai_r, const std::vector<float>& kai_i,
                                   const std::vector<ne10_fft_cpx_float32_t>& ne10_out) {
    const size_t m_half = m / 2 + 1;
    double max_err = 0.0;
    double sum_sq = 0.0;
    const size_t bins = n * m_half;

    for (size_t r = 0; r < n; ++r) {
        for (size_t c = 0; c < m_half; ++c) {
            const size_t nidx = r * m_half + c;
            const size_t kidx = r * m + c;
            const double dr = static_cast<double>(ne10_out[nidx].r) - static_cast<double>(kai_r[kidx]);
            const double di = static_cast<double>(ne10_out[nidx].i) - static_cast<double>(kai_i[kidx]);
            const double err = std::sqrt(dr * dr + di * di);

            if (err > max_err) {
                max_err = err;
            }
            sum_sq += dr * dr + di * di;
        }
    }

    CompareStats s{};
    s.max_err = max_err;
    s.rmse = std::sqrt(sum_sq / static_cast<double>(bins));
    return s;
}

template <typename Fn>
double time_average_ms(int reps, Fn&& fn) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        fn();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const auto elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return elapsed_ms / static_cast<double>(reps);
}

bool run_case(size_t n, size_t m, int reps, CaseResult* out) {
    const size_t m_half = m / 2 + 1;

    std::vector<float> x(n * m);
    std::vector<float> f_nr(n * n), f_ni(n * n);
    std::vector<float> f_mr(m * m), f_mi(m * m);
    std::vector<float> kai_r(n * m, 0.0f), kai_i(n * m, 0.0f);

    fill_random(x.size(), x.data(), static_cast<uint32_t>(n * 131u + m * 17u + 1u));
    generate_dft_matrix(n, f_nr.data(), f_ni.data());
    generate_dft_matrix(m, f_mr.data(), f_mi.data());

    ne10_fft_r2c_cfg_float32_t cfg_r2c = ne10_fft_alloc_r2c_float32(static_cast<ne10_int32_t>(m));
    ne10_fft_cfg_float32_t cfg_c2c = ne10_fft_alloc_c2c_float32(static_cast<ne10_int32_t>(n));
    if (cfg_r2c == nullptr || cfg_c2c == nullptr) {
        if (cfg_r2c != nullptr) {
            ne10_fft_destroy_r2c_float32(cfg_r2c);
        }
        if (cfg_c2c != nullptr) {
            ne10_fft_destroy_c2c_float32(cfg_c2c);
        }
        std::cerr << "Ne10 FFT config allocation failed for size " << n << "x" << m << "\n";
        return false;
    }

    std::vector<ne10_fft_cpx_float32_t> ne10_out(n * m_half);
    std::vector<ne10_fft_cpx_float32_t> col_buf(n);

    run_dft_2d(n, m, x.data(), f_nr.data(), f_ni.data(), f_mr.data(), f_mi.data(), kai_r.data(), kai_i.data());
    run_ne10_2d_r2c_fft(n, m, m_half, x.data(), ne10_out.data(), cfg_r2c, cfg_c2c, col_buf.data());

    const double kai_ms = time_average_ms(reps, [&] {
        run_dft_2d(n, m, x.data(), f_nr.data(), f_ni.data(), f_mr.data(), f_mi.data(), kai_r.data(), kai_i.data());
    });

    const double ne10_ms = time_average_ms(reps, [&] {
        run_ne10_2d_r2c_fft(n, m, m_half, x.data(), ne10_out.data(), cfg_r2c, cfg_c2c, col_buf.data());
    });

    const CompareStats stats = compare_half_spectrum(n, m, kai_r, kai_i, ne10_out);

    ne10_fft_destroy_r2c_float32(cfg_r2c);
    ne10_fft_destroy_c2c_float32(cfg_c2c);

    const double n_half = static_cast<double>(n / 2 + 1);
    const double kai_macs = 2.0 * n_half * static_cast<double>(n) * static_cast<double>(m) +
                            3.0 * n_half * static_cast<double>(m) * static_cast<double>(m);
    const double kai_flops = 2.0 * kai_macs;

    const double fft_flops = 5.0 * static_cast<double>(n) * static_cast<double>(m) *
                             std::log2(static_cast<double>(n * m));

    out->n = n;
    out->m = m;
    out->kai_ms = kai_ms;
    out->ne10_ms = ne10_ms;
    out->kai_gflops = kai_flops / (kai_ms * 1.0e6);
    out->ne10_gflops = fft_flops / (ne10_ms * 1.0e6);
    out->err = stats;

    std::cout << "\n--- Size " << n << "x" << m << " ---\n";
    std::cout << "NE10 FFT : " << std::fixed << std::setprecision(3) << ne10_ms << " ms, " << std::setprecision(2)
              << out->ne10_gflops << " GF/s\n";
    std::cout << "KAI DFT  : " << std::fixed << std::setprecision(3) << kai_ms << " ms, " << std::setprecision(2)
              << out->kai_gflops << " GF/s\n";
    std::cout << "Speedup (NE10 over KAI): " << std::setprecision(2) << (kai_ms / ne10_ms) << "x\n";
    std::cout << "Error (KAI vs NE10 half-spectrum): max=" << std::scientific << std::setprecision(3) << stats.max_err
              << ", rmse=" << stats.rmse << "\n";

    return true;
}

void print_summary(const std::vector<CaseResult>& results) {
    std::cout << "\n=== Scaling Summary ===\n";
    std::cout << std::left << std::setw(10) << "Size" << std::right << std::setw(12) << "ms(NE10)" << std::setw(12)
              << "ms(KAI)" << std::setw(14) << "GF/s(NE10)" << std::setw(14) << "GF/s(KAI)" << std::setw(14)
              << "NE10/KAI" << "\n";

    for (const auto& r : results) {
        const std::string size = std::to_string(r.n) + "x" + std::to_string(r.m);
        std::cout << std::left << std::setw(10) << size << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << r.ne10_ms << std::setw(12) << r.kai_ms << std::setprecision(2)
                  << std::setw(14) << r.ne10_gflops << std::setw(14) << r.kai_gflops << std::setw(14)
                  << (r.kai_ms / r.ne10_ms) << "x\n";
    }

    std::cout << "\nFootnote:\n";
    std::cout << "GF/s = estimated_FLOPs / (time_ms * 1e6)\n";
    std::cout << "KAI estimated_FLOPs = 2 * [2*(N/2+1)*N*M + 3*(N/2+1)*M*M]\n";
    std::cout << "NE10 estimated_FLOPs = 5 * N * M * log2(N*M)\n";
}

}  // namespace

int main() {
    if (ne10_init() != NE10_OK) {
        std::cerr << "Ne10 init failed\n";
        return 1;
    }

    std::cout << "DFT via KleidiAI SME GEMMs vs Ne10 FFT (float32)\n";
    std::cout << "Comparing half-spectrum: N x (M/2 + 1) bins\n";
    std::cout << "Timing: average over " << kReps << " reps\n";

    std::vector<std::pair<size_t, size_t>> sizes;
    for (size_t dim = 16; dim <= 4096; dim <<= 1) {
        sizes.emplace_back(dim, dim);
    }

    bool all_ok = true;
    std::vector<CaseResult> results;
    results.reserve(sizes.size());

    for (const auto& size : sizes) {
        CaseResult r{};
        if (!run_case(size.first, size.second, kReps, &r)) {
            all_ok = false;
            continue;
        }
        results.push_back(r);
    }

    print_summary(results);

    std::cout << "\nOverall: " << (all_ok ? "PASS" : "FAIL") << "\n";
    return all_ok ? 0 : 1;
}

#endif
