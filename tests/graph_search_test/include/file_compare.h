/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 T-Head (Shanghai) Semiconductor Co., Ltd. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef FILE_COMPARE_H_
#define FILE_COMPARE_H_

#include <fstream>
#include <string>
#include <streambuf>

bool CompareFiles(const std::string& file_path1, const std::string& file_path2) {
    std::ifstream file1(file_path1);
    std::ifstream file2(file_path2);

    // File stream error checking omitted for brevity

    std::string content1((std::istreambuf_iterator<char>(file1)), std::istreambuf_iterator<char>());
    std::string content2((std::istreambuf_iterator<char>(file2)), std::istreambuf_iterator<char>());

    return content1 == content2;
}

#endif  // FILE_COMPARE_H_