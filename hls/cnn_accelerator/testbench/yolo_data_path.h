#ifndef YOLO_TESTBENCH_DATA_PATH_H
#define YOLO_TESTBENCH_DATA_PATH_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

// CNN testbench 共用的数据目录定位逻辑，不参与综合。
inline std::string yolo_test_data_dir() {
    // 优先使用编译器记录的原始头文件路径，例如仓库外手工创建的工程。
    std::string source = __FILE__;
    std::replace(source.begin(), source.end(), '\\', '/');
    const std::string suffix = "hls/cnn_accelerator/testbench/yolo_data_path.h";
    if (source.size() >= suffix.size()) {
        const auto offset = source.size() - suffix.size();
        if ((offset == 0 || source[offset - 1] == '/') &&
            source.compare(offset, suffix.size(), suffix) == 0 &&
            std::ifstream(source).good()) {
            return source.substr(0, offset) + "data/yolov2_tiny";
        }
    }
    // Vitis HLS 可能复制或改写测试源路径。对仓库内的工程，
    // 从 csim/build 或 sim 工作目录向上查找源文件以识别仓库。
    std::string prefix;
    for (int depth = 0; depth <= 16; ++depth) {
        if (std::ifstream(prefix + suffix).good() &&
            std::ifstream(prefix + "hls/cnn_accelerator/script/hls_ddr.tcl").good()) {
            return prefix + "data/yolov2_tiny";
        }
        prefix += "../";
    }
    std::cerr << "无法定位仓库的数据目录；请通过 hls_ddr.tcl 启动测试，"
                 "或使用仓库内的仿真工程及原始 testbench。\n";
    return {};
}

#endif
