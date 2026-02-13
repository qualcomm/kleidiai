#!/bin/bash

TOOLS_DIR=tools
#mkdir -p ${TOOLS_DIR}
cd ${TOOLS_DIR}

# Official LLVM 18.1.8 prebuilt binary (Linux x86_64, Ubuntu 18.04)
URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz"

echo "Downloading clang+llvm 18.1.8..."
wget ${URL}

echo "Extracting..."
tar -xf clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04.tar.xz

echo "Done. Extracted into ${TOOLS_DIR}/clang+llvm-18.1.8-x86_64-linux-gnu-ubuntu-18.04"

