#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(matmul_clamp_qai8_qai8p_qsi8cxp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    enable_language(ASM_MARMASM)
else()
    enable_language(ASM)
endif()

set(KLEIDIAI_PATH ../../)
set(MATMUL_PACK_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/pack/)
set(MATMUL_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/)
set(KAI_COMMON_PATH ${KLEIDIAI_PATH}/kai/)

# KleidiAI include directories
include_directories(
    ${KLEIDIAI_PATH}
    ${MATMUL_PACK_PATH}
    ${MATMUL_PATH})

# Files requires to build the executable
add_executable(matmul_clamp_qai8_qai8p_qsi8cxp
    matmul_clamp_qai8_qai8p_qsi8cxp.cpp
    ${MATMUL_PACK_PATH}/kai_lhs_pack_x8p2vlx4_x8_sme.c
    ${MATMUL_PACK_PATH}/kai_lhs_pack_x8p2vlx4_x8_sme_asm.S
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme_asm.S
    ${MATMUL_PATH}/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa.c
    ${MATMUL_PATH}/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa_asm.S
    ${KAI_COMMON_PATH}/kai_common_sme_asm.S
)

target_compile_options(matmul_clamp_qai8_qai8p_qsi8cxp
    PRIVATE -march=armv8.2-a+sve+sve2
)

set_source_files_properties(
    ${MATMUL_PATH}/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa.c
    PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+sve+sve2")

set_source_files_properties(
    ${MATMUL_PATH}/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_qmx_mopa_asm.S
    ${MATMUL_PACK_PATH}/kai_lhs_pack_x8p2vlx4_x8_sme.c
    ${MATMUL_PACK_PATH}/kai_lhs_pack_x8p2vlx4_x8_sme_asm.S
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.c
    ${MATMUL_PACK_PATH}/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme_asm.S
    ${KAI_COMMON_PATH}/kai_common_sme_asm.S
    PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+sve+sve2")

target_compile_definitions(matmul_clamp_qai8_qai8p_qsi8cxp
    PRIVATE $<$<CONFIG:Debug>:KAI_DEBUG>)
