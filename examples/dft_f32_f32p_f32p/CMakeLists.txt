#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.16)

project(dft_f32_f32p_f32p LANGUAGES C CXX ASM)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

# Ne10 dependency (fetched and built as an embedded dependency).
set(NE10_GIT_REPOSITORY "https://github.com/projectNe10/Ne10.git" CACHE STRING "Ne10 Git repository")
set(NE10_GIT_TAG "master" CACHE STRING "Ne10 Git tag/branch/commit")
set(NE10_PATCH_FILE "" CACHE FILEPATH "Optional patch file to apply on fetched Ne10")

# Configure Ne10 for static library use from this example.
set(NE10_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(NE10_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(NE10_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NE10_BUILD_UNIT_TEST OFF CACHE BOOL "" FORCE)
set(NE10_ENABLE_MATH OFF CACHE BOOL "" FORCE)
set(NE10_ENABLE_DSP ON CACHE BOOL "" FORCE)
set(NE10_ENABLE_IMGPROC OFF CACHE BOOL "" FORCE)
set(NE10_ENABLE_PHYSICS OFF CACHE BOOL "" FORCE)

if(ANDROID)
    set(NE10_ANDROID_TARGET_ARCH aarch64 CACHE STRING "" FORCE)
    if(NOT DEFINED ANDROID_API_LEVEL)
        set(ANDROID_API_LEVEL 30 CACHE STRING "" FORCE)
    endif()
else()
    set(GNULINUX_PLATFORM ON CACHE BOOL "" FORCE)
    set(NE10_LINUX_TARGET_ARCH aarch64 CACHE STRING "" FORCE)
endif()

FetchContent_Declare(
    ne10
    GIT_REPOSITORY ${NE10_GIT_REPOSITORY}
    GIT_TAG ${NE10_GIT_TAG}
)

FetchContent_GetProperties(ne10)
if(NOT ne10_POPULATED)
    FetchContent_Populate(ne10)
endif()

if(NE10_PATCH_FILE)
    if(NOT EXISTS "${NE10_PATCH_FILE}")
        message(FATAL_ERROR "NE10_PATCH_FILE='${NE10_PATCH_FILE}' was not found")
    endif()

    find_program(GIT_EXECUTABLE git REQUIRED)

    execute_process(
        COMMAND ${GIT_EXECUTABLE} apply --check "${NE10_PATCH_FILE}"
        WORKING_DIRECTORY "${ne10_SOURCE_DIR}"
        RESULT_VARIABLE patch_check_result
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(patch_check_result EQUAL 0)
        message(STATUS "Applying Ne10 patch: ${NE10_PATCH_FILE}")
        execute_process(
            COMMAND ${GIT_EXECUTABLE} apply "${NE10_PATCH_FILE}"
            WORKING_DIRECTORY "${ne10_SOURCE_DIR}"
            RESULT_VARIABLE patch_apply_result
        )
        if(NOT patch_apply_result EQUAL 0)
            message(FATAL_ERROR "Failed to apply NE10_PATCH_FILE='${NE10_PATCH_FILE}'")
        endif()
        message(STATUS "Ne10 patch applied successfully")
    else()
        execute_process(
            COMMAND ${GIT_EXECUTABLE} apply --reverse --check "${NE10_PATCH_FILE}"
            WORKING_DIRECTORY "${ne10_SOURCE_DIR}"
            RESULT_VARIABLE patch_reverse_check_result
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(patch_reverse_check_result EQUAL 0)
            message(STATUS "Ne10 patch already applied: ${NE10_PATCH_FILE}")
        else()
            message(FATAL_ERROR "NE10_PATCH_FILE='${NE10_PATCH_FILE}' does not apply to fetched Ne10")
        endif()
    endif()
endif()

add_subdirectory("${ne10_SOURCE_DIR}" "${ne10_BINARY_DIR}")

set(KLEIDIAI_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../)
set(MATMUL_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p)
set(PACK_PATH ${KLEIDIAI_PATH}/kai/ukernels/matmul/pack)

add_executable(dft_f32_f32p_f32p
    dft_f32_f32p_f32p.cpp
    ${KLEIDIAI_PATH}/kai/kai_common_sme_asm.S
    ${MATMUL_PATH}/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.c
    ${MATMUL_PATH}/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa_asm.S
    ${PACK_PATH}/kai_lhs_pack_f32p2vlx1_f32_sme.c
    ${PACK_PATH}/kai_lhs_pack_f32p2vlx1_f32_sme_asm.S
    ${PACK_PATH}/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.c
    ${PACK_PATH}/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme_asm.S
)

target_include_directories(dft_f32_f32p_f32p PRIVATE
    ${KLEIDIAI_PATH}
    ${MATMUL_PATH}
    ${PACK_PATH}
    ${ne10_SOURCE_DIR}/inc
)

target_compile_options(dft_f32_f32p_f32p PRIVATE
    -march=armv9.2-a+sme
)

target_link_libraries(dft_f32_f32p_f32p PRIVATE
    NE10
    m
)
