#
# SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# + Changes from Qualcomm Technologies, Inc. are provided under the following license:
# + Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# + SPDX-License-Identifier: BSD-3-Clause-Clear
#

cmake_minimum_required(VERSION 3.16)

project(KleidiAI
    VERSION 1.12.0
    LANGUAGES C CXX
)

if(MSVC)
    enable_language(ASM_MARMASM)
else()
    enable_language(ASM)
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(GNUInstallDirs)

option(KLEIDIAI_BUILD_TESTS       "Build unit tests."             ON)
option(KLEIDIAI_BUILD_BENCHMARK   "Build the benchmark tool."     ON)
option(KLEIDIAI_ENABLE_CLANG_TIDY "Build with Clang-Tidy checks." OFF)
option(MLAS_ENABLE_KLEIDIAI_QMX_COEXIST "Enable both Arm Kleidiai and Qualcomm QMX library for MLAS" OFF)

if(KLEIDIAI_ENABLE_CLANG_TIDY)
    set(CMAKE_C_CLANG_TIDY "clang-tidy")
    set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
endif()

if(MSVC)
    set(KLEIDIAI_WARNING_FLAGS_BASE
        # "/Wall" - This flag is disabled until the kernel library is cleaned up.
    )
else()
    set(KLEIDIAI_WARNING_FLAGS_BASE
        "-Wall"
        "-Wdisabled-optimization"
        "-Wextra"
        "-Wformat-security"
        "-Wformat=2"
        "-Winit-self"
        "-Wstrict-overflow=2"
        "-Wswitch-default"
        "-Wcast-qual"
    )

    # C only flags not present in C++
    set(KLEIDIAI_WARNING_FLAGS_C
        "-Wmissing-prototypes"
        "-Wstrict-prototypes"
    )

    set(KLEIDIAI_WARNING_FLAGS_CXX
        "-Wctor-dtor-privacy"
        "-Woverloaded-virtual"
        "-Wsign-promo"
        "-Wmissing-declarations"
    )
endif()

set(KLEIDIAI_WARNING_FLAGS
    ${KLEIDIAI_WARNING_FLAGS_BASE}
    $<$<COMPILE_LANGUAGE:C>:${KLEIDIAI_WARNING_FLAGS_C}>
    $<$<COMPILE_LANGUAGE:CXX>:${KLEIDIAI_WARNING_FLAGS_CXX}>
)

set(KLEIDIAI_MIN_CLANG_VERSION 11)
set(KLEIDIAI_MIN_GNU_VERSION 11)

if(CMAKE_C_COMPILER_ID STREQUAL "Clang" AND CMAKE_C_COMPILER_VERSION VERSION_LESS ${KLEIDIAI_MIN_CLANG_VERSION})
    message(WARNING "KleidiAI: Using non-supported Clang version. Expected ${KLEIDIAI_MIN_CLANG_VERSION} or newer, received ${CMAKE_C_COMPILER_VERSION}.")
endif()

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS ${KLEIDIAI_MIN_GNU_VERSION})
    message(WARNING "KleidiAI: Using non-supported GCC version. Expected ${KLEIDIAI_MIN_GNU_VERSION} or newer, received ${CMAKE_C_COMPILER_VERSION}.")
endif()

if(NOT MLAS_ENABLE_KLEIDIAI_QMX_COEXIST)
	set(KLEIDIAI_FILES_SCALAR
	)

	set(KLEIDIAI_FILES_NEON_FP16
	)

	set(KLEIDIAI_FILES_NEON_FP16_DOTPROD_ASM
	)

	set(KLEIDIAI_FILES_NEON_FP16_DOTPROD
		${KLEIDIAI_FILES_NEON_FP16_DOTPROD_ASM}
	)
	set(KLEIDIAI_FILES_NEON_FP16_I8MM_ASM

	)

	set(KLEIDIAI_FILES_NEON_FP16_I8MM
		${KLEIDIAI_FILES_NEON_FP16_I8MM_ASM}
	)

	set(KLEIDIAI_FILES_NEON_BF16

	)

	set(KLEIDIAI_FILES_NEON_FP16_BF16

	)

	set(KLEIDIAI_FILES_NEON_ASM

	)

	set(KLEIDIAI_FILES_NEON
		${KLEIDIAI_FILES_NEON_ASM}

	)

	set(KLEIDIAI_FILES_NEON_DOTPROD_ASM

	)

	set(KLEIDIAI_FILES_NEON_DOTPROD
		${KLEIDIAI_FILES_NEON_DOTPROD_ASM}

	)

	set(KLEIDIAI_FILES_NEON_I8MM_ASM

	)

	set(KLEIDIAI_FILES_NEON_I8MM
		${KLEIDIAI_FILES_NEON_I8MM_ASM}

	)

	set(KLEIDIAI_FILES_SME_ASM
		kai/kai_common_sme_asm.S
		kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa.c
		kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme_mopa_asm.S
		kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme.c
		kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x16p2vlx2_x16p_sme_asm.S
		kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme.c
		kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x32p2vlx1_x32p_sme_asm.S
		kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme.c
		kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme_asm.S
		kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme.c
		kai/ukernels/matmul/pack/kai_lhs_pack_f32p2vlx1_f32_sme_asm.S
		kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme.c
		kai/ukernels/matmul/pack/kai_lhs_pack_x16p2vlx2_x16_sme_asm.S
		kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme.c
		kai/ukernels/matmul/pack/kai_lhs_pack_x8p2vlx4_x8_sme_asm.S
		kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.c
		kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme_asm.S
		kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme.c
		kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x16p2vlx2b_x16_x16_sme_asm.S
		kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme.c
		kai/ukernels/matmul/pack/kai_rhs_imatmul_pack_kxn_x32p2vlx1b_x32_x32_sme_asm.S
		kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme.c
		kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p16vlx1b_f32_f32_sme_asm.S
		kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme.c
		kai/ukernels/matmul/pack/kai_rhs_pack_kxn_f32p2vlx1biasf32_f32_f32_sme_asm.S
		kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme.c
		kai/ukernels/matmul/pack/kai_rhs_pack_kxn_qsi8cxp2vlx4sb_qs8cx_f32_i32_sme_asm.S
		kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme.c
		kai/ukernels/matmul/pack/kai_rhs_pack_kxn_x16p2vlx2b_x16_x16_sme_asm.S
		kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme.c
		kai/ukernels/matmul/pack/kai_rhs_pack_nxk_f32p2vlx1biasf32_f32_f32_sme_asm.S
		kai/ukernels/matmul/pack/kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme.c
		kai/ukernels/matmul/pack/kai_rhs_pack_nxk_x16p2vlx2b_x16_x16_sme_asm.S

		kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme1_mopa.c
		kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme1_mopa_asm.S
		kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme1_mopa.c
		kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme1_mopa_asm.S

		kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa.c
		kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa_asm.S
		kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa.c
		kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa_asm.S

		kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa.c
		kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme_mopa_asm.S
		kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme1_mopa.c
		kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme1_mopa_asm.S

	)

	set(KLEIDIAI_FILES_SME

		${KLEIDIAI_FILES_SME_ASM}
	)

	set(KLEIDIAI_FILES_SME2_ASM
		kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa.c
		kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme2_mopa_asm.S
		kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.c
		kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa_asm.S
		kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.c
		kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa_asm.S
		kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.c
		kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa_asm.S
		kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.c
		kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa_asm.S
	)

	set(KLEIDIAI_FILES_SME2
		${KLEIDIAI_FILES_SME2_ASM}
	)

	add_library(kleidiai)
	add_library(${PROJECT_NAME}::kleidiai ALIAS kleidiai)

	target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_SCALAR})

	# Selectively enable architecture features.
	if(NOT MSVC)
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON})
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_FP16})
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_FP16_DOTPROD})
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_FP16_I8MM})
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_BF16})
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_FP16_BF16})
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_DOTPROD})
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_I8MM})
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_SME})
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_SME2})

		set_source_files_properties(${KLEIDIAI_FILES_SCALAR}            PROPERTIES COMPILE_OPTIONS -march=armv8-a${KLEIDIAI_INTERNAL_EXTRA_ARCH})
		set_source_files_properties(${KLEIDIAI_FILES_NEON}              PROPERTIES COMPILE_OPTIONS -march=armv8-a${KLEIDIAI_INTERNAL_EXTRA_ARCH})
		set_source_files_properties(${KLEIDIAI_FILES_NEON_FP16}         PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+fp16${KLEIDIAI_INTERNAL_EXTRA_ARCH})
		set_source_files_properties(${KLEIDIAI_FILES_NEON_FP16_DOTPROD} PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+fp16+dotprod${KLEIDIAI_INTERNAL_EXTRA_ARCH})
		set_source_files_properties(${KLEIDIAI_FILES_NEON_FP16_I8MM}    PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+fp16+i8mm${KLEIDIAI_INTERNAL_EXTRA_ARCH})
		set_source_files_properties(${KLEIDIAI_FILES_NEON_BF16}         PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+bf16${KLEIDIAI_INTERNAL_EXTRA_ARCH})
		set_source_files_properties(${KLEIDIAI_FILES_NEON_FP16_BF16}    PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+bf16+fp16${KLEIDIAI_INTERNAL_EXTRA_ARCH})
		set_source_files_properties(${KLEIDIAI_FILES_NEON_DOTPROD}      PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+dotprod${KLEIDIAI_INTERNAL_EXTRA_ARCH})
		set_source_files_properties(${KLEIDIAI_FILES_NEON_I8MM}         PROPERTIES COMPILE_OPTIONS -march=armv8.2-a+i8mm${KLEIDIAI_INTERNAL_EXTRA_ARCH})

		# Use -fno-tree-vectorize option to disable compiler based vectorization
		set_source_files_properties(${KLEIDIAI_FILES_SME}          PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+sve+sve2+sme${KLEIDIAI_INTERNAL_EXTRA_ARCH}")
		set_source_files_properties(${KLEIDIAI_FILES_SME2}         PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+sve+sve2${KLEIDIAI_INTERNAL_EXTRA_ARCH}")
	else()
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_ASM})
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_DOTPROD_ASM})
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_NEON_I8MM_ASM})
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_SME_ASM})
		target_sources(kleidiai PRIVATE ${KLEIDIAI_FILES_SME2_ASM})

		set_source_files_properties(${KLEIDIAI_FILES_SCALAR}            PROPERTIES COMPILE_OPTIONS /arch:armv8.0${KLEIDIAI_INTERNAL_EXTRA_ARCH})
		set_source_files_properties(${KLEIDIAI_FILES_NEON_I8MM_ASM}     PROPERTIES COMPILE_OPTIONS /arch:armv8.2${KLEIDIAI_INTERNAL_EXTRA_ARCH})
		set_source_files_properties(${KLEIDIAI_FILES_NEON_DOTPROD_ASM}  PROPERTIES COMPILE_OPTIONS /arch:armv8.2${KLEIDIAI_INTERNAL_EXTRA_ARCH})
		set_source_files_properties(${KLEIDIAI_FILES_NEON_ASM}          PROPERTIES COMPILE_OPTIONS /arch:armv8.2${KLEIDIAI_INTERNAL_EXTRA_ARCH})
		set_source_files_properties(${KLEIDIAI_FILES_SME_ASM}           PROPERTIES COMPILE_OPTIONS /arch:armv8.2${KLEIDIAI_INTERNAL_EXTRA_ARCH})
		set_source_files_properties(${KLEIDIAI_FILES_SME2_ASM}          PROPERTIES COMPILE_OPTIONS /arch:armv8.2${KLEIDIAI_INTERNAL_EXTRA_ARCH})

		set(KLEIDIAI_FILES_ASM
			${KLEIDIAI_FILES_SME_ASM}
			${KLEIDIAI_FILES_SME2_ASM}
			${KLEIDIAI_FILES_NEON_ASM}
			${KLEIDIAI_FILES_NEON_DOTPROD_ASM}
			${KLEIDIAI_FILES_NEON_I8MM_ASM})
		list(FILTER KLEIDIAI_FILES_ASM INCLUDE REGEX "^.*\.S$")

		set_source_files_properties(${KLEIDIAI_FILES_ASM} PROPERTIES LANGUAGE ASM_MARMASM)
	endif()

	target_include_directories(kleidiai PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kai/>
	)

	target_compile_options(kleidiai
		PRIVATE $<$<NOT:$<BOOL:${MSVC}>>:-Wpedantic>
		PRIVATE ${KLEIDIAI_WARNING_FLAGS}
	)

	# Micro-kernel library does not support link time optimization.
	#
	# The linker does not respect the use of -fno-tree-vectorize when building
	# SME kernels. LTO can only be enabled once there is no need for disabling
	# auto-vectorization.
	set_target_properties(kleidiai PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)


	install(TARGETS kleidiai
		EXPORT ${PROJECT_NAME}Targets
		RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
		LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
		ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	)

	install(EXPORT ${PROJECT_NAME}Targets
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
		NAMESPACE ${PROJECT_NAME}::
	)

	install(DIRECTORY kai
		DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
		FILES_MATCHING
			PATTERN *.h
	)

	include(CMakePackageConfigHelpers)

	write_basic_package_version_file(
		"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
		VERSION ${PROJECT_VERSION}
		COMPATIBILITY SameMajorVersion
	)

	set(ConfigContents "@PACKAGE_INIT@\n
	include(\"\${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}Targets.cmake\")\n"
	)

	file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake.in"
		"${ConfigContents}"
	)

	configure_package_config_file("${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake.in"
		"${PROJECT_NAME}Config.cmake"
		INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
	)

	install(FILES
		"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
		"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
	)
else()

	set(KLEIDIAI_FILES_SME_ASM
		#kai/kai_common_sme_asm.S

		kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme1_mopa.c
		kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme1_mopa_asm.S
		kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme1_mopa.c
		kai/ukernels/matmul/imatmul_clamp_qai8_qai8p_qsi8cxp/kai_imatmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme1_mopa_asm.S

		kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa.c
		kai/ukernels/matmul/imatmul_clamp_f32_f32p_f32p/kai_imatmul_clamp_f32_f32p2vlx1_f32p2vlx1b_2vlx2vl_sme1_mopa_asm.S
		kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa.c
		kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme1_mopa_asm.S
		kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme1_mopa.c
		kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme1_mopa_asm.S	

	)

	set(KLEIDIAI_FILES_SME
		kai/ukernels/matmul/pack/kai_lhs_pack_bf16p2vlx2_f32_sme.c
		kai/ukernels/matmul/pack/kai_rhs_pack_kxn_bf16p2vlx2b_f32_x32_sme.c
		${KLEIDIAI_FILES_SME_ASM}
	)

	add_library(kleidiai-qmx)
	add_library(${PROJECT_NAME}::kleidiai-qmx ALIAS kleidiai-qmx)

	# Selectively enable architecture features.
	if(NOT MSVC)
		target_sources(kleidiai-qmx PRIVATE ${KLEIDIAI_FILES_SME})

		# Use -fno-tree-vectorize option to disable compiler based vectorization
		set_source_files_properties(${KLEIDIAI_FILES_SME}          PROPERTIES COMPILE_OPTIONS "-fno-tree-vectorize;-march=armv8.2-a+sve+sve2+sme${KLEIDIAI_INTERNAL_EXTRA_ARCH}")
	else()
		target_sources(kleidiai-qmx PRIVATE ${KLEIDIAI_FILES_SME_ASM})

		set_source_files_properties(${KLEIDIAI_FILES_SME_ASM}           PROPERTIES COMPILE_OPTIONS /arch:armv8.2${KLEIDIAI_INTERNAL_EXTRA_ARCH})

		set(KLEIDIAI_FILES_ASM
			${KLEIDIAI_FILES_SME_ASM})
		list(FILTER KLEIDIAI_FILES_ASM INCLUDE REGEX "^.*\.S$")

		set_source_files_properties(${KLEIDIAI_FILES_ASM} PROPERTIES LANGUAGE ASM_MARMASM)
	endif()

	target_include_directories(kleidiai-qmx PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
		$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/kai/>
	)

	target_compile_options(kleidiai-qmx
		PRIVATE $<$<NOT:$<BOOL:${MSVC}>>:-Wpedantic>
		PRIVATE ${KLEIDIAI_WARNING_FLAGS}
	)

	# Micro-kernel library does not support link time optimization.
	#
	# The linker does not respect the use of -fno-tree-vectorize when building
	# SME kernels. LTO can only be enabled once there is no need for disabling
	# auto-vectorization.
	set_target_properties(kleidiai-qmx PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

	target_link_libraries(kleidiai-qmx PRIVATE kleidiai)

	install(TARGETS kleidiai-qmx
		EXPORT ${PROJECT_NAME}Targets
		RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
		LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
		ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	)

	install(EXPORT ${PROJECT_NAME}Targets
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
		NAMESPACE ${PROJECT_NAME}::
	)

	install(DIRECTORY kai
		DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
		FILES_MATCHING
			PATTERN *.h
	)

	include(CMakePackageConfigHelpers)

	write_basic_package_version_file(
		"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
		VERSION ${PROJECT_VERSION}
		COMPATIBILITY SameMajorVersion
	)

	set(ConfigContents "@PACKAGE_INIT@\n
	include(\"\${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}Targets.cmake\")\n"
	)

	file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake.in"
		"${ConfigContents}"
	)

	configure_package_config_file("${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake.in"
		"${PROJECT_NAME}Config.cmake"
		INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
	)

	install(FILES
		"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
		"${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
	)
endif() # MLAS_ENABLE_KLEIDIAI_QMX_COEXIST
