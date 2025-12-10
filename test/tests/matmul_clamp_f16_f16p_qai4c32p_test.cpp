//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>
#include <tuple>

#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_qai4c32p/kai_matmul_clamp_f16_f16p1vlx2_qai4c32p4vlx2_1vlx4vl_sme2_mopa.h"
#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_qai4c32p/kai_matmul_clamp_f16_f16p_qai4c32p_interface.h"
#include "kai/ukernels/matmul/pack/kai_lhs_pack_f16pmrx2_f32_neon.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon.h"
#include "test/common/abi_checker.hpp"
#include "test/common/buffer.hpp"
#include "test/common/compare.hpp"
#include "test/common/cpu_info.hpp"
#include "test/common/data_format.hpp"
#include "test/common/float16.hpp"
#include "test/common/int4.hpp"
#include "test/common/matmul_test_common.hpp"
#include "test/common/matrix_portion.hpp"
#include "test/common/memory.hpp"
#include "test/common/round.hpp"
#include "test/common/test_suite.hpp"
#include "test/reference/cast.hpp"
#include "test/reference/clamp.hpp"
#include "test/reference/fill.hpp"
#include "test/reference/matmul.hpp"
#include "test/reference/pack.hpp"
#include "test/reference/quantize.hpp"

namespace kai::test {

// Interface for the LHS and RHS packed size and packing micro-kernels
using kai_get_lhs_packed_size_func_t = decltype(&kai_get_lhs_packed_size_lhs_pack_f16pmrx2_f32_neon);
using kai_get_rhs_packed_size_func_t =
    decltype(&kai_get_rhs_packed_size_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon);
using kai_get_lhs_packed_offset_func_t = decltype(&kai_get_lhs_packed_offset_lhs_pack_f16pmrx2_f32_neon);
using kai_get_rhs_packed_offset_func_t =
    decltype(&kai_get_rhs_packed_offset_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon);
using kai_get_lhs_offset_func_t = decltype(&kai_get_lhs_offset_lhs_pack_f16pmrx2_f32_neon);
using kai_get_rhs_offset_func_t =
    decltype(&kai_get_rhs_offset_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon);
using kai_run_lhs_pack_func_t = decltype(&kai_run_lhs_pack_f16pmrx2_f32_neon);
using kai_run_rhs_pack_func_t =
    decltype(&kai_run_rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon);

// Micro-kernel interface
struct kai_qai4c32p_pack_functions {
    kai_get_rhs_packed_size_func_t packed_size;
    kai_get_rhs_packed_offset_func_t get_packed_offset;
    kai_get_rhs_offset_func_t get_offset;
    kai_run_rhs_pack_func_t run_pack;
};

struct kai_f16p_pack_functions {
    kai_get_lhs_packed_size_func_t packed_size;
    kai_get_lhs_packed_offset_func_t get_packed_offset;
    kai_get_lhs_offset_func_t get_offset;
    kai_run_lhs_pack_func_t run_pack;
};

static const std::array<
    UkernelMatmulPackVariant<
        kai_matmul_clamp_f16_f16p_qai4c32p_ukernel, kai_f16p_pack_functions, kai_qai4c32p_pack_functions>,
    1>
    variants_kai_matmul_clamp_f16_f16p_qai4c32p = {
        {UKERNEL_MATMUL_PACK_VARIANT(
             clamp_f16_f16p1vlx2_qai4c32p4vlx2_1vlx4vl_sme2_mopa, cpu_has_sme, lhs_pack_f16pmrx2_f32_neon,
             rhs_pack_nxk_qai4c32ps1s0nrx4_qau4c32s0s1_f32_f32_f32_neon, true)}};

static const auto test_matmul_shapes = testing::Values(
    MatMulShape{32, 32, 32}
);

static const auto test_portions = testing::Values(
    MatrixPortion(0, 0, 1, 1)
);

static const auto test_block_lengths = testing::Values(32);

// Executes the LHS packing micro-kernel.
static inline Buffer pack_lhs_f16p(
    const kai_f16p_pack_functions& pack_interface, size_t M, size_t K, size_t bl, size_t mr, size_t kr, size_t sr,
    const Buffer& lhs_f32, size_t stride, size_t rect_start_row, size_t rect_height) {
    const auto imp_packed_lhs_size = pack_interface.packed_size(M, K, bl, mr, kr, sr);
    Buffer imp_packed_lhs(imp_packed_lhs_size, 0);

    auto lhs_offset = pack_interface.get_offset(rect_start_row, stride);
    auto lhs_packed_offset = pack_interface.get_packed_offset(rect_start_row, K, bl, mr, kr, sr);

    abi_check(
        pack_interface.run_pack, rect_height, K, bl, mr, kr, sr, 0,
        reinterpret_cast<const float*>(lhs_f32.data() + lhs_offset), stride, imp_packed_lhs.data() + lhs_packed_offset);

    return (imp_packed_lhs);
}

// Executes the RHS packing micro-kernel.
static inline Buffer pack_rhs_qai4c32p(
    const kai_qai4c32p_pack_functions& pack_interface, size_t N, size_t K, size_t bl, size_t nr, size_t kr, size_t sr,
    const Buffer& rhs_values_qai4, const bool has_bias, const Buffer& biases, const Buffer& rhs_scales,
    const Buffer& rhs_zp, bool s0s1_input) {
    const auto imp_packed_rhs_size = pack_interface.packed_size(N, K, nr, kr, bl);
    Buffer imp_packed_rhs(imp_packed_rhs_size);

    // Runs the RHS packing micro-kernel.
    kai_rhs_pack_nxk_qai4c32p_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 8;

    // Cast to unsigned int
    auto rhs_qau4s1s0 = cast_qsu4_qsi4(rhs_values_qai4.data(), N * K);

    abi_check(
        pack_interface.run_pack, 1, N, K, nr, kr, sr, bl,
        reinterpret_cast<const uint8_t*>(s0s1_input ? convert_s0s1_s1s0(rhs_qau4s1s0).data() : rhs_qau4s1s0.data()),
        rhs_zp.data(), has_bias ? biases.data() : nullptr, rhs_scales.data(), imp_packed_rhs.data(), 0, &params);

    return (imp_packed_rhs);
}

class MatMulTest_f16_f16p_qai4c32p : public ::testing::TestWithParam<MatMulTestPortionedParamsWithBias_WithBL> {};

TEST_P(MatMulTest_f16_f16p_qai4c32p, EndToEnd) {
    const auto& [variant_index, matmul_shape, bl, portion, has_bias] = GetParam();
    const auto& ukernel_variant = variants_kai_matmul_clamp_f16_f16p_qai4c32p.at(variant_index);

    if (ukernel_variant.ukernel.fn_is_supported && !ukernel_variant.ukernel.fn_is_supported()) {
        GTEST_SKIP() << "Unsupported CPU feature";
    }

    const std::uint32_t seed = 0;

    const size_t M = matmul_shape.m;
    const size_t N = matmul_shape.n;
    const size_t K = matmul_shape.k;

    if (K % bl != 0) {
        GTEST_SKIP() << "K must be a multiple of bl";
    }

    const auto mr = ukernel_variant.ukernel.interface.get_mr();
    const auto nr = ukernel_variant.ukernel.interface.get_nr();
    const auto kr = ukernel_variant.ukernel.interface.get_kr();
    const auto sr = ukernel_variant.ukernel.interface.get_sr();

    if (mr == 1 && M > 1) {
        GTEST_SKIP() << "Kernel does not support M != 1";
    }

    auto m_step = ukernel_variant.ukernel.interface.get_m_step();
    ASSERT_TRUE(m_step % mr == 0);

    auto n_step = ukernel_variant.ukernel.interface.get_n_step();
    ASSERT_TRUE(n_step % nr == 0);

    const auto rect = portion.compute_portion(M, N, m_step, n_step);
    if (rect.height() == 0 || rect.width() == 0) {
        GTEST_SKIP() << "Empty dimension of matrix(" << rect.width() << "," << rect.height() << ")";
    }

    // Generates input data.
    const auto ref_lhs = fill_random<float>(M * K, seed + 0);
    const auto ref_rhs = fill_random<float>(N * K, seed + 1);
    Buffer ref_biases;

    if (has_bias) {
        ref_biases = fill_random<float>(N, seed + 2);
    }

    // Runs the reference implementation.
    //   * Converts LHS to FP16
    //   * Quantizes the RHS matrix using 4-bit asymmetric quantization.
    //   * Performs GEMM.
    const auto ref_lhs_f16 = cast<Float16, float>(ref_lhs.data(), M * K);
    const auto [ref_rhs_qai4, ref_rhs_scales, ref_rhs_zero_points] =
        quantize_asymmetric_per_block_dynamic<float, Int4, float, int32_t>(ref_rhs.data(), N, K, bl);

    const auto ref_dst_no_clamp =
        matmul_nt_t_quantized<Float16, float, int32_t, Int4, float, int32_t, float, float, int32_t, float>(
            M, N, K, ref_lhs_f16.data(), nullptr, nullptr, 1, bl, ref_rhs_qai4.data(), ref_rhs_scales.data(),
            ref_rhs_zero_points.data(), 1, bl, has_bias ? ref_biases.data() : nullptr, nullptr, nullptr, 1);

    // Clamps the reference output.
    const auto clamp_ratio = 0.8F;
    const auto [clamp_min, clamp_max] = find_clamp_range<float>(ref_dst_no_clamp.data(), M * N, clamp_ratio);
    const auto ref_dst_float = clamp<float>(ref_dst_no_clamp.data(), M * N, clamp_min, clamp_max);

    // Cast the reference output to F16
    auto ref_dst = cast<Float16, float>(ref_dst_float.data(), ref_dst_float.size() * 8 / size_in_bits<float>);

    // Runs the LHS packing micro-kernel.
    const auto lhs_start_row = rect.start_row();
    auto imp_packed_lhs = pack_lhs_f16p(
        ukernel_variant.lhs_pack_interface, M, K, bl, mr, kr, sr, ref_lhs, K * sizeof(float), lhs_start_row,
        rect.height());
    auto lhs_packed_offset = ukernel_variant.lhs_pack_interface.get_packed_offset(lhs_start_row, K, bl, mr, kr, sr);
    auto lhs_matmul_offset = ukernel_variant.ukernel.interface.get_lhs_packed_offset(lhs_start_row, K, bl);

    ASSERT_EQ(lhs_packed_offset, lhs_matmul_offset);

    // Prepare the offsets as the RHS packing micro-kernel expects the scaled zero-points in float.
    const size_t num_blocks_per_row = round_up_division(K, bl);
    const size_t ref_zp_size = N * num_blocks_per_row;
    const size_t ref_zp_size_in_bytes = ref_zp_size * sizeof(float);
    Buffer ref_rhs_zp_f32(ref_zp_size_in_bytes);
    for (size_t i = 0; i < ref_zp_size; ++i) {
        reinterpret_cast<float*>(ref_rhs_zp_f32.data())[i] =
            -reinterpret_cast<const int32_t*>(ref_rhs_zero_points.data())[i] *
            reinterpret_cast<const float*>(ref_rhs_scales.data())[i];
    }

    const auto rhs_start_row = rect.start_col();
    auto imp_packed_rhs = pack_rhs_qai4c32p(
        ukernel_variant.rhs_pack_interface, N, K, bl, nr, kr, sr, ref_rhs_qai4, has_bias, ref_biases, ref_rhs_scales,
        ref_rhs_zp_f32, ukernel_variant.rhs_s0s1_input);
    auto rhs_packed_offset = ukernel_variant.rhs_pack_interface.get_packed_offset(rhs_start_row, K, nr, kr, bl);
    auto rhs_matmul_offset = ukernel_variant.ukernel.interface.get_rhs_packed_offset(rhs_start_row, K, bl);
    ASSERT_EQ(rhs_packed_offset, rhs_matmul_offset);

    const auto dst_stride_row = N * sizeof(uint16_t);
    const auto dst_stride_col = sizeof(uint16_t);
    const auto dst_offset =
        ukernel_variant.ukernel.interface.get_dst_offset(rect.start_row(), rect.start_col(), dst_stride_row);
    const auto ref_dst_offset = rect.start_row() * dst_stride_row + rect.start_col() * dst_stride_col;
    ASSERT_EQ(dst_offset, ref_dst_offset);

    // Runs the GEMM micro-kernel.
    const auto imp_dst_size = ukernel_variant.ukernel.interface.get_dst_size(M, N);
    ASSERT_EQ(imp_dst_size, ref_dst.size());
    Buffer imp_dst(imp_dst_size);
    abi_check(
        ukernel_variant.ukernel.interface.run_matmul, rect.height(), rect.width(), K, bl,
        imp_packed_lhs.data() + lhs_matmul_offset, imp_packed_rhs.data() + rhs_matmul_offset,
        reinterpret_cast<float*>(imp_dst.data() + dst_offset), dst_stride_row, dst_stride_col, clamp_min, clamp_max);

    // Compares the output of the micro-kernels against the output of the reference implementation for the portion
    // tested.
    DefaultMismatchHandler handler(0, 0.1, 0, 0.05);
    DataFormat dst_format = DataFormat(DataType::FP16);
    const auto success = compare(imp_dst.data(), ref_dst.data(), dst_format, M, N, rect, handler);
    ASSERT_TRUE(success);
}

INSTANTIATE_TEST_SUITE_P(
    MatMul, MatMulTest_f16_f16p_qai4c32p,
    testing::Combine(
        testing::Range<size_t>(0, variants_kai_matmul_clamp_f16_f16p_qai4c32p.size()), test_matmul_shapes,
        test_block_lengths, test_portions, testing::Bool()),
    [](const auto& info) {
        const auto variant_idx = std::get<0>(info.param);
        const std::string name{variants_kai_matmul_clamp_f16_f16p_qai4c32p.at(variant_idx).ukernel.name};
        const auto shape = std::get<MatMulShape>(info.param);
        const auto bl = std::get<2>(info.param);
        const auto portion = std::get<3>(info.param);
        const auto has_bias = std::get<4>(info.param);

        std::ostringstream sstream;
        sstream << name << "__Variant_" << variant_idx << "__";
        PrintTo(shape, &sstream);
        sstream << "__BL_" << bl << "_";
        if (has_bias) {
            sstream << "_withBias";
        } else {
            sstream << "_noBias";
        }
        if (variants_kai_matmul_clamp_f16_f16p_qai4c32p.at(variant_idx).rhs_s0s1_input) {
            sstream << "_RHS_s0s1__";
        } else {
            sstream << "_RHS_s1s0__";
        }
        PrintTo(portion, &sstream);

        return sstream.str();
    });

}  // namespace kai::test
