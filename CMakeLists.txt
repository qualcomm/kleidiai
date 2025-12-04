#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required(VERSION 3.16)

project(KleidiAI-Qmx
    VERSION 1.16.0
    LANGUAGES C CXX
)


set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")





set(KLEIDIAI_MIN_CLANG_VERSION 11)
set(KLEIDIAI_MIN_GNU_VERSION 11)



set(KLEIDIAI_FILES_SME_ASM



    kai/ukernels/matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme1_dot.c
    kai/ukernels/matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme1_dot_asm.S

    kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme1_mopa.c
    kai/ukernels/matmul/matmul_clamp_fp32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p2vlx2_bf16p2vlx2_2vlx2vl_sme1_mopa_asm.S
    
    kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme1_mopa.c
    kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme1_mopa_asm.S
    kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme1_dot.c
    kai/ukernels/matmul/matmul_clamp_f16_qsi8d32p_qai4c32p/kai_matmul_clamp_f16_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme1_dot_asm.S
    
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme1_mopa.c
    kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme1_sdot.c
    
    kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qsi4c32p4vlx4_1vlx4vl_sme1_mopa.c
    kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qsi4c32p4vlx4_1x4vl_sme1_sdot.c
    
    kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme1_mopa.c
    kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1vlx4_qai4c32p4vlx4_1vlx4vl_sme1_mopa_asm.S
    kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme1_dot.c
    kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qai4c32p/kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p4vlx4_1x4vl_sme1_dot_asm.S
    
    
    kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme1.c
)

set(KLEIDIAI_FILES_SME
    ${KLEIDIAI_FILES_SME_ASM}
)

set(KLEIDIAI_FILES_SME2_ASM

)

set(KLEIDIAI_FILES_SME2
    ${KLEIDIAI_FILES_SME2_ASM}


)

add_library(kleidiai_qmx)
add_library(${PROJECT_NAME}::kleidiai_qmx ALIAS kleidiai_qmx)

# Selectively enable architecture features.
if(NOT MSVC)

    target_sources(kleidiai_qmx PRIVATE ${KLEIDIAI_FILES_SME})
    target_sources(kleidiai_qmx PRIVATE ${KLEIDIAI_FILES_SME2})
    set_source_files_properties(${KLEIDIAI_FILES_SME}          PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+sve+sve2+sme${KLEIDIAI_INTERNAL_EXTRA_ARCH}")
    set_source_files_properties(${KLEIDIAI_FILES_SME2}         PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+sve+sve2${KLEIDIAI_INTERNAL_EXTRA_ARCH}")
else()

    target_sources(kleidiai_qmx PRIVATE ${KLEIDIAI_FILES_SME_ASM})
    target_sources(kleidiai_qmx PRIVATE ${KLEIDIAI_FILES_SME2_ASM})
    set_source_files_properties(${KLEIDIAI_FILES_SME_ASM}                PROPERTIES COMPILE_OPTIONS /arch:armv8.2${KLEIDIAI_INTERNAL_EXTRA_ARCH})
    set_source_files_properties(${KLEIDIAI_FILES_SME2_ASM}               PROPERTIES COMPILE_OPTIONS /arch:armv8.2${KLEIDIAI_INTERNAL_EXTRA_ARCH})

    set(KLEIDIAI_FILES_ASM
        ${KLEIDIAI_FILES_SME_ASM}
        ${KLEIDIAI_FILES_SME2_ASM} )

    list(FILTER KLEIDIAI_FILES_ASM INCLUDE REGEX "^.*\.S$")

    set_source_files_properties(${KLEIDIAI_FILES_ASM} PROPERTIES LANGUAGE ASM_MARMASM)
endif()

target_include_directories(kleidiai_qmx PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kai/>

)


target_compile_options(kleidiai_qmx
    PRIVATE $<$<NOT:$<BOOL:${MSVC}>>:-Wpedantic>
    PRIVATE ${KLEIDIAI_WARNING_FLAGS}
)

# Micro-kernel library does not support link time optimization.

set_target_properties(kleidiai_qmx PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

