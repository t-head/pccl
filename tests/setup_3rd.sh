#!/bin/bash
#*************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#*************************************************************************


set -euo pipefail
INSTALL_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/3rd_party
DOWNLOAD_PATH=${INSTALL_PATH}/download

# create install dir
if [ ! -d "${INSTALL_PATH}" ]
then
    mkdir -p ${INSTALL_PATH}
fi

# create download dir
if [ ! -d "${DOWNLOAD_PATH}" ]
then
    mkdir -p ${DOWNLOAD_PATH}
fi

setup_gtest() {
    GTEST_PATH=${INSTALL_PATH}/gtest

    if [ -s "${GTEST_PATH}/lib/libgtest.a" ]
    then
        echo "GTest installed!"
        return
    fi

    echo
    echo -------- GTest Installing --------

    GTEST_ZIP_FILE=${DOWNLOAD_PATH}/gtest-1.7.0.zip
    GTEST_SRC_FILE=https://github.com/google/googletest/archive/refs/tags/release-1.7.0.zip

    cd ${DOWNLOAD_PATH}
    if [ ! -f ${GTEST_ZIP_FILE} ]
    then
        wget ${GTEST_SRC_FILE} -O ${GTEST_ZIP_FILE}
    fi

    unzip -o ${GTEST_ZIP_FILE}

    GTEST_SRC_PATH=${DOWNLOAD_PATH}/googletest-release-1.7.0
    chmod +x ${GTEST_SRC_PATH}

    cd ${GTEST_SRC_PATH}

    rm -rf build
    mkdir build
    cd build
    cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
    make -j16

    if [ ! -d "${GTEST_PATH}" ]
    then
        mkdir -p ${GTEST_PATH}
    fi

    cp -rf ${GTEST_SRC_PATH}/include/ ${GTEST_PATH}

    GTEST_LIB_PATH=${GTEST_PATH}/lib
    if [ ! -d "${GTEST_LIB_PATH}" ]
    then
        mkdir -p ${GTEST_LIB_PATH}
    fi
    cp -rf ${GTEST_SRC_PATH}/build/libgtest.a ${GTEST_LIB_PATH}

    make clean
    cmake -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" -DCMAKE_POSITION_INDEPENDENT_CODE=ON .
    make -j16
    cp -rf ${GTEST_SRC_PATH}/build/libgtest.a ${GTEST_LIB_PATH}/libgtest_no_cxx11_abi.a

    echo
    echo -------- Complete GTest Installing --------
}

echo
echo ======== Start Installing 3rd-part Components =====
setup_gtest
echo ======== End Installing 3rd-part Components =======
echo
