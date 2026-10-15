//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon.h"

namespace kai::test {
namespace {

constexpr size_t kK = 32;
constexpr size_t kKr = 8;
constexpr size_t kSr = 2;
constexpr uint8_t kGuard = 0x5A;

void set_qsu2(std::vector<uint8_t>& rhs, size_t rhs_stride, size_t n, size_t k, uint8_t value) {
    const size_t byte_idx = (n * rhs_stride) + (k / 4);
    const size_t shift = (k % 4) * 2;
    rhs[byte_idx] = (rhs[byte_idx] & ~(0x03U << shift)) | ((value & 0x03U) << shift);
}

uint8_t get_qsu2(const std::vector<uint8_t>& rhs, size_t rhs_stride, size_t n, size_t k) {
    return (rhs[(n * rhs_stride) + (k / 4)] >> ((k % 4) * 2)) & 0x03;
}

uint8_t get_regrouped_byte(
    const std::vector<uint8_t>& rhs, size_t rhs_stride, size_t row, size_t k_unit, size_t k_in_group) {
    uint8_t value = 0;
    for (size_t field = 0; field < 4; ++field) {
        const size_t k = (k_unit * 16) + (field * 4) + k_in_group;
        value |= (get_qsu2(rhs, rhs_stride, row, k) ^ 0x02) << (2 * field);
    }
    return value;
}

void run_packer(
    size_t n, size_t k, size_t nr, const std::vector<uint8_t>& rhs, const std::vector<float>& bias,
    const std::vector<float>& scale, uint8_t* packed) {
    kai_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon_params params{};
    params.lhs_zero_point = 1;
    params.rhs_zero_point = 2;

    kai_run_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(
        1, n, k, nr, kKr, kSr, rhs.data(), bias.empty() ? nullptr : bias.data(), scale.data(), packed, 0, &params,
        nullptr);
}

TEST(RhsPackNxkQsi2cxp4vlx4s12s8s4s0Qsu2cxNeon, PacksRequestedFourByFourLayout) {
    constexpr size_t n = 4;
    constexpr size_t nr = 4;
    constexpr size_t rhs_stride = kK / 4;
    std::vector<uint8_t> rhs(n * rhs_stride, 0xAA);

    const std::array<std::array<uint8_t, 4>, 4> values = {{{0, 1, 2, 3}, {1, 2, 3, 0}, {2, 3, 0, 1}, {3, 0, 1, 2}}};
    for (size_t field = 0; field < 4; ++field) {
        for (size_t k_in_group = 0; k_in_group < 4; ++k_in_group) {
            set_qsu2(rhs, rhs_stride, 0, (field * 4) + k_in_group, values.at(field).at(k_in_group));
            set_qsu2(rhs, rhs_stride, 0, 16 + (field * 4) + k_in_group, values.at(3 - field).at(k_in_group));
        }
    }

    const size_t packed_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(n, kK, nr, kKr, kSr);
    std::vector<uint8_t> packed(packed_size, kGuard);
    const std::vector<float> scale(n, 1.0F);
    const std::vector<float> bias(n, 0.0F);
    run_packer(n, kK, nr, rhs, bias, scale, packed.data());

    // Row 0, K=0..15 becomes [q12 q8 q4 q0], [q13 q9 q5 q1], ... after q ^ 2 recoding.
    EXPECT_EQ(packed[0], 0x4E);
    EXPECT_EQ(packed[1], 0x93);
    EXPECT_EQ(packed[2], 0xE4);
    EXPECT_EQ(packed[3], 0x39);
    for (size_t k_in_group = 0; k_in_group < 4; ++k_in_group) {
        EXPECT_EQ(packed[(4 * nr) + k_in_group], get_regrouped_byte(rhs, rhs_stride, 0, 1, k_in_group));
    }
    for (size_t offset = 0; offset < nr * kK / 4; ++offset) {
        const bool belongs_to_row_zero = offset < 4 || (offset >= 4 * nr && offset < (4 * nr) + 4);
        if (!belongs_to_row_zero) {
            EXPECT_EQ(packed[offset], 0);
        }
    }
}

TEST(RhsPackNxkQsi2cxp4vlx4s12s8s4s0Qsu2cxNeon, RegroupsKWithinEachRow) {
    constexpr size_t n = 8;
    constexpr size_t nr = 8;
    constexpr size_t rhs_stride = kK / 4;
    std::vector<uint8_t> rhs(n * rhs_stride, 0);

    for (size_t row = 0; row < n; ++row) {
        for (size_t k = 0; k < kK; ++k) {
            set_qsu2(rhs, rhs_stride, row, k, static_cast<uint8_t>((row + (3 * k) + (k / 4)) & 0x03));
        }
    }

    const size_t packed_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(n, kK, nr, kKr, kSr);
    std::vector<uint8_t> packed(packed_size, kGuard);
    const std::vector<float> scale(n, 1.0F);
    run_packer(n, kK, nr, rhs, {}, scale, packed.data());

    for (size_t k_unit = 0; k_unit < kK / 16; ++k_unit) {
        for (size_t row = 0; row < nr; ++row) {
            for (size_t k_in_group = 0; k_in_group < 4; ++k_in_group) {
                const size_t offset = (k_unit * 4 * nr) + (4 * row) + k_in_group;
                EXPECT_EQ(packed[offset], get_regrouped_byte(rhs, rhs_stride, row, k_unit, k_in_group))
                    << "k_unit=" << k_unit << ", row=" << row << ", k_in_group=" << k_in_group;
            }
        }
    }
}

TEST(RhsPackNxkQsi2cxp4vlx4s12s8s4s0Qsu2cxNeon, AdvancesAcrossFourKUnits) {
    constexpr size_t k = 64;
    constexpr size_t n = 8;
    constexpr size_t nr = 8;
    constexpr size_t rhs_stride = k / 4;
    std::vector<uint8_t> rhs(n * rhs_stride, 0);

    for (size_t row = 0; row < n; ++row) {
        for (size_t k_idx = 0; k_idx < k; ++k_idx) {
            const uint8_t value = static_cast<uint8_t>((row + k_idx + (k_idx / 4) + (k_idx / 16)) & 0x03);
            set_qsu2(rhs, rhs_stride, row, k_idx, value);
        }
    }

    const size_t packed_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(n, k, nr, kKr, kSr);
    std::vector<uint8_t> packed(packed_size, kGuard);
    const std::vector<float> scale(n, 1.0F);
    run_packer(n, k, nr, rhs, {}, scale, packed.data());

    for (size_t k_unit = 0; k_unit < k / 16; ++k_unit) {
        for (size_t row = 0; row < nr; ++row) {
            for (size_t k_in_group = 0; k_in_group < 4; ++k_in_group) {
                const size_t offset = (k_unit * 4 * nr) + (4 * row) + k_in_group;
                EXPECT_EQ(packed[offset], get_regrouped_byte(rhs, rhs_stride, row, k_unit, k_in_group));
            }
        }
    }
}

TEST(RhsPackNxkQsi2cxp4vlx4s12s8s4s0Qsu2cxNeon, PadsTailsAndStoresMetadata) {
    constexpr size_t n = 5;
    constexpr size_t nr = 8;
    constexpr size_t rhs_stride = kK / 4;
    constexpr size_t guard_size = 16;
    std::vector<uint8_t> rhs(n * rhs_stride, 0);
    std::vector<int32_t> expected_sums(n, 0);

    for (size_t row = 0; row < n; ++row) {
        for (size_t k = 0; k < kK; ++k) {
            const uint8_t value = static_cast<uint8_t>((row + k) & 0x03);
            set_qsu2(rhs, rhs_stride, row, k, value);
            expected_sums[row] += static_cast<int32_t>(value) - 2;
        }
    }

    const std::vector<float> scale = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    const std::vector<float> bias = {-1.0F, -2.0F, -3.0F, -4.0F, -5.0F};
    const size_t payload_size = nr * kK / 4;
    const size_t packed_size =
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(n, kK, nr, kKr, kSr);
    std::vector<uint8_t> storage(packed_size + (2 * guard_size), kGuard);
    uint8_t* const packed = storage.data() + guard_size;
    run_packer(n, kK, nr, rhs, bias, scale, packed);

    for (size_t k_unit = 0; k_unit < kK / 16; ++k_unit) {
        for (size_t row = 0; row < nr; ++row) {
            for (size_t k_in_group = 0; k_in_group < 4; ++k_in_group) {
                const size_t offset = (k_unit * 4 * nr) + (4 * row) + k_in_group;
                const uint8_t expected = row < n
                    ? get_regrouped_byte(rhs, rhs_stride, row, k_unit, k_in_group)
                    : 0;
                EXPECT_EQ(packed[offset], expected);
            }
        }
    }

    const auto* sums = reinterpret_cast<const int32_t*>(packed + payload_size);
    const auto* packed_scales = reinterpret_cast<const float*>(sums + nr);
    const auto* packed_biases = packed_scales + nr;
    for (size_t row = 0; row < nr; ++row) {
        if (row < n) {
            EXPECT_EQ(sums[row], expected_sums[row]);
            EXPECT_EQ(packed_scales[row], scale[row]);
            EXPECT_EQ(packed_biases[row], bias[row]);
        } else {
            EXPECT_EQ(sums[row], 0);
            EXPECT_EQ(packed_scales[row], 0.0F);
            EXPECT_EQ(packed_biases[row], 0.0F);
        }
    }

    EXPECT_TRUE(std::all_of(storage.begin(), storage.begin() + guard_size, [](uint8_t value) {
        return value == kGuard;
    }));
    EXPECT_TRUE(std::all_of(storage.end() - guard_size, storage.end(), [](uint8_t value) {
        return value == kGuard;
    }));
}

TEST(RhsPackNxkQsi2cxp4vlx4s12s8s4s0Qsu2cxNeon, ComputesSizesAndOffsets) {
    constexpr size_t n = 17;
    constexpr size_t nr = 8;
    constexpr size_t expected_stride = nr * ((kK / 4) + sizeof(int32_t) + sizeof(float) + sizeof(float));

    EXPECT_EQ(kai_get_n_step_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(nr), nr);
    EXPECT_EQ(kai_get_rhs_offset_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(2 * nr, kK / 4), 2 * nr * kK / 4);
    EXPECT_EQ(
        kai_get_rhs_packed_stride_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(kK, nr, kKr, kSr), expected_stride);
    EXPECT_EQ(
        kai_get_rhs_packed_offset_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(2 * nr, kK, nr, kKr, kSr),
        2 * expected_stride);
    EXPECT_EQ(
        kai_get_rhs_packed_size_rhs_pack_nxk_qsi2cxp4vlx4s12s8s4s0_qsu2cx_neon(n, kK, nr, kKr, kSr),
        3 * expected_stride);
}

}  // namespace
}  // namespace kai::test
