#ifndef YOLO_TESTBENCH_DATA_PATH_H
#define YOLO_TESTBENCH_DATA_PATH_H

#include <cstdlib>
#include <iostream>
#include <string>

// 仅供软件测试使用；默认目录不依赖 csim/build 的当前工作目录。
inline std::string yolo_test_data_dir() {
    const char* override_dir = std::getenv("YOLO_DATA_DIR");
    if (override_dir && override_dir[0]) return override_dir;
#ifdef YOLO_DEFAULT_DATA_DIR
    // HLS 脚本只向测试文件传入此绝对路径，不改变 DUT 的编译选项。
    return YOLO_DEFAULT_DATA_DIR;
#else
    // 手工 HLS 工程可能保留相对源码路径；先从运行目录规范化真实源码路径。
    std::string source = __FILE__;
    if (source.empty() || source[0] != '/') {
        char* absolute_source = ::realpath(source.c_str(), nullptr);
        if (absolute_source) {
            source = absolute_source;
            std::free(absolute_source);
        }
    }
    if (source.empty() || source[0] != '/') {
        std::cerr << "无法由源码路径定位默认数据；请以绝对路径添加测试源文件，"
                     "或设置 YOLO_DATA_DIR / YOLO_DEFAULT_DATA_DIR。\n";
        return std::string();
    }
    for (int level = 0; level < 4; ++level) {
        const std::string::size_type slash = source.find_last_of('/');
        if (slash == std::string::npos) return std::string();
        source.resize(slash);
    }
    return source + "/data/yolov2_tiny";
#endif
}

#endif
