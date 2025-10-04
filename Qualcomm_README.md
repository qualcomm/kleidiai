<!--
       Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
       SPDX-License-Identifier: BSD-3-Clause-Clear
-->

# KleidiAI Build Instructions for Qualcomm devices

This repository contains build scripts and toolchain configuration for building the KleidiAI project targeting Armv8.5-A architecture with SME and SVE2 extensions for Qualcomm devices

---

## 📋 Prerequisites

Ensure the following tools are installed and properly configured:

- **Arm GNU Toolchain**  
  Download from Arm Developer

- **Clang 16.0.0**  
  Required for cross-compilation targeting `aarch64-linux-gnu`

- **CMake**  
  Used for project configuration and build system generation

---

## 📦 Cloning the Repository

      git clone https://github.com/qualcomm/kleidiai.git
      cd kleidiai
      git checkout kleidi-ai-qmx

## ⚙️ Build Instructions
### Cross-compiling on Linux
1) Ensure the toolchain file is located at:
 
      cmake/toolchains/aarch64-none-linux-gnu.toolchain.cmake

2) Set the environment variable to point to the Clang toolchain path:
   
      CMakeset(CROSS_COMPILE_LLVM_PATH /prj/qct/chips/swarch/tools/clang-16.0.0)Show more lines

3) Run the build script:
     
     ./run.sh

