#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TOOLCHAIN_FILE="${REPO_ROOT}/cmake/toolchains/aarch64-none-linux-gnu.toolchain.cmake"
LLVM_DIR="${REPO_ROOT}/tools/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04"
NDK_VERSION="${NDK_VERSION:-r27d}"
NDK_ZIP="android-ndk-${NDK_VERSION}-linux.zip"
NDK_URL="${NDK_URL:-https://dl.google.com/android/repository/${NDK_ZIP}}"
NDK_DIR="${REPO_ROOT}/tools/android-ndk-${NDK_VERSION}"
NDK_LINK="${REPO_ROOT}/tools/android-ndk"
BINARY_NAME="dft_f32_f32p_f32p"
BINARY_DST_DIR="$(eval echo ~/../binaries)"

# Optional environment overrides:
#   BUILD_TYPE=Release|Debug
#   TARGET_OS=linux|android
#   NE10_GIT_REPOSITORY=<git-url>
#   NE10_GIT_TAG=<tag|branch|commit>
#   NE10_PATCH_FILE=<path-to-ne10.patch>
#   ANDROID_API=30
#   ANDROID_ABI=arm64-v8a
#   RUN_ON_DEVICE=0|1
#   ANDROID_NDK=<path-to-ndk> (optional; auto-downloaded to tools/ if missing)
#
# Usage:
#   ./run.sh
#   ./run.sh clean

BUILD_TYPE="${BUILD_TYPE:-Release}"
TARGET_OS="${TARGET_OS:-android}"
NE10_GIT_REPOSITORY="${NE10_GIT_REPOSITORY:-https://github.com/projectNe10/Ne10.git}"
NE10_GIT_TAG="${NE10_GIT_TAG:-master}"
NE10_PATCH_FILE="${NE10_PATCH_FILE:-}"
ANDROID_API="${ANDROID_API:-30}"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
RUN_ON_DEVICE="${RUN_ON_DEVICE:-0}"
ANDROID_NDK="${ANDROID_NDK:-}"

if [[ "${TARGET_OS}" == "android" ]]; then
    BUILD_DIR="${SCRIPT_DIR}/build-android"
fi

if [[ "${1:-}" == "clean" ]]; then
    echo "Cleaning ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

cmake_args=(
    -S "${SCRIPT_DIR}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DNE10_GIT_REPOSITORY="${NE10_GIT_REPOSITORY}"
    -DNE10_GIT_TAG="${NE10_GIT_TAG}"
)

if [[ -n "${NE10_PATCH_FILE}" ]]; then
    cmake_args+=(-DNE10_PATCH_FILE="${NE10_PATCH_FILE}")
fi

if [[ "${TARGET_OS}" == "android" ]]; then
    if [[ -z "${ANDROID_NDK}" ]]; then
        if [[ -d "${NDK_LINK}" ]]; then
            ANDROID_NDK="${NDK_LINK}"
        elif [[ -d "${NDK_DIR}" ]]; then
            ANDROID_NDK="${NDK_DIR}"
        else
            echo "Android NDK not found. Downloading ${NDK_ZIP} into ${REPO_ROOT}/tools ..."
            mkdir -p "${REPO_ROOT}/tools"
            (
                cd "${REPO_ROOT}/tools"
                wget -c "${NDK_URL}" -O "${NDK_ZIP}"
                unzip -q -o "${NDK_ZIP}"
                ln -sfn "android-ndk-${NDK_VERSION}" "android-ndk"
            )
            ANDROID_NDK="${NDK_LINK}"
        fi
    fi

    if [[ ! -f "${ANDROID_NDK}/build/cmake/android.toolchain.cmake" ]]; then
        echo "ERROR: invalid ANDROID_NDK path: ${ANDROID_NDK}"
        exit 1
    fi

    cmake_args+=(
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake"
        -DANDROID_ABI="${ANDROID_ABI}"
        -DANDROID_PLATFORM="android-${ANDROID_API}"
        -DANDROID_API_LEVEL="${ANDROID_API}"
    )
    echo "Configuring (Android ${ANDROID_ABI}, API ${ANDROID_API})..."
else
    if [[ ! -d "${LLVM_DIR}" ]]; then
        echo "Required LLVM toolchain is missing at: ${LLVM_DIR}"
        echo "Bootstrapping tools via ${REPO_ROOT}/get_requirements_tools.sh ..."
        (
            cd "${REPO_ROOT}"
            ./get_requirements_tools.sh
        )
    fi

    if [[ ! -d "${LLVM_DIR}" ]]; then
        echo "ERROR: LLVM toolchain bootstrap failed; expected directory was not created."
        exit 1
    fi

    if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
        echo "ERROR: toolchain file not found: ${TOOLCHAIN_FILE}"
        exit 1
    fi

    cmake_args+=(-DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}")
    echo "Configuring (linux-aarch64 via ru.sh-style toolchain)..."
fi

cmake "${cmake_args[@]}"

echo "Building..."
cmake --build "${BUILD_DIR}" --parallel

mkdir -p "${BINARY_DST_DIR}"
cp -f "${BUILD_DIR}/${BINARY_NAME}" "${BINARY_DST_DIR}/"

if [[ "${TARGET_OS}" == "android" && "${RUN_ON_DEVICE}" == "1" ]]; then
    if adb get-state >/dev/null 2>&1; then
        DEVICE_DIR="/data/local/tmp/dft_f32_f32p_f32p"
        adb shell "mkdir -p ${DEVICE_DIR}"
        adb push "${BUILD_DIR}/${BINARY_NAME}" "${DEVICE_DIR}/${BINARY_NAME}" >/dev/null
        adb shell "chmod +x ${DEVICE_DIR}/${BINARY_NAME}"
        echo ""
        echo "Running on connected Android device..."
        adb shell "${DEVICE_DIR}/${BINARY_NAME}"
    else
        echo ""
        echo "No Android device detected via adb; skipped device run."
    fi
fi

echo ""
echo "Built binary: ${BUILD_DIR}/${BINARY_NAME}"
echo "Copied binary: ${BINARY_DST_DIR}/${BINARY_NAME}"
if [[ "${TARGET_OS}" == "android" ]]; then
    echo "Run on an arm64 Android device (SME-capable for KAI SME path)."
else
    echo "Run on an AArch64 SME-capable Linux target."
fi
