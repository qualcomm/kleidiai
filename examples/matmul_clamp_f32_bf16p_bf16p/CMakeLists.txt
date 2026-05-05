#
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(KleidiAI)

set(CMAKE_CXX_STANDARD 17)
set(KLEIDIAI_PATH ../../)
set(MATMUL_PACK_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/pack/)
set(MATMUL_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/matmul_clamp_f32_bf16p_bf16p/)
set(MATMUL_QMX_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/)
set(KAI_COMMON_PATH ${KLEIDIAI_PATH}/kai/)

enable_language(ASM)

# KleidiAI include directories
include_directories(
    ${KLEIDIAI_PATH}
    ${MATMUL_PACK_PATH}
    ${MATMUL_QMX_PATH}
    ${MATMUL_PATH})

# Files requires to build the executable
add_executable(matmul_clamp_f32_bf16p_bf16p
    matmul_clamp_f32_bf16p_bf16p.cpp
    ${MATMUL_PATH}/kai_matmul_clamp_f32_bf16p1x4_bf16p12x4b_1x36_neon_dot.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.c
    ${MATMUL_PACK_PATH}/kai_lhs_quant_pack_bf16p1x4_f32_neon.c
    ${MATMUL_PACK_PATH}/kai_lhs_quant_pack_bf16p8x4_f32_neon.c
    ${MATMUL_PACK_PATH}/kai_rhs_quant_pack_kxn_bf16p12x4biasf32_f32_neon.c
    ${MATMUL_QMX_PATH}/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa.c
    ${MATMUL_QMX_PATH}/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa_asm.S
    ${MATMUL_PACK_PATH}/kai_lhs_pack_bf16p2vlx2_f32_sme.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.c
    ${KAI_COMMON_PATH}/kai_common_sme_asm.S
)

target_compile_options(matmul_clamp_f32_bf16p_bf16p
    PRIVATE -march=armv8.2-a+bf16
)

set_source_files_properties(
    ${MATMUL_QMX_PATH}/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa.c
    PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+sve+sve2+bf16")

set_source_files_properties(
    ${MATMUL_QMX_PATH}/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_qmx_mopa_asm.S
    PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+sve+sve2+bf16")

set_source_files_properties(
    ${MATMUL_PACK_PATH}/kai_lhs_pack_bf16p2vlx2_f32_sme.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.c
    ${KAI_COMMON_PATH}/kai_common_sme_asm.S
    PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+sve+sve2")

target_compile_definitions(matmul_clamp_f32_bf16p_bf16p
    PRIVATE $<$<CONFIG:Debug>:KAI_DEBUG>
)
