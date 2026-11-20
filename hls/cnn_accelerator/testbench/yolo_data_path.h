#ifndef YOLO_TESTBENCH_DATA_PATH_H
#define YOLO_TESTBENCH_DATA_PATH_H

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>

// CNN testbench 共用的数据目录定位逻辑，不参与综合。
inline std::string yolo_test_data_dir() {
#ifdef YOLO_DEFAULT_DATA_DIR
    // HLS 脚本只向测试文件传入此绝对路径，不改变 DUT 的编译选项。
    return YOLO_DEFAULT_DATA_DIR;
#else
    // 手工工程使用原头文件的路径。若编译器记录的是相对路径，
    // 则该路径在仿真工作目录中也必须可读。
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
    std::cerr << "无法定位数据目录：请用仓库中的原始 testbench 源码，"
                 "或在测试编译选项中指定 YOLO_DEFAULT_DATA_DIR。\n";
    return {};
#endif
}

#endif
