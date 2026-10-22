#!/usr/bin/env python3
"""从训练导出的metadata生成少量标量；无依赖，无像素LUT，不修改训练工程。"""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path(
        "/home/liaoyingkai/python_project/minist_train_fpga/CNN_FPGA/generated/metadata.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "vitis/cnn_accel_cam/src/camera_model_input.h")
    args = parser.parse_args()
    model_input = json.loads(args.metadata.read_text())["model_input"]
    scale = model_input["scale"]
    zero = model_input["zero_point"]
    if isinstance(zero, list) and len(zero) == 1:
        zero = zero[0]
    if isinstance(scale, bool) or not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0:
        parser.error("model_input.scale必须为有限正数")
    if isinstance(zero, bool) or not isinstance(zero, (float, int)) or zero != 0:
        parser.error("仅支持model_input.zero_point=0（或单元素[0]）")
    args.output.write_text(
        "/* 自动生成；来源: " + str(args.metadata.resolve()) + "\n"
        " * scale保持导出double精度；mean/std固定对应dataset.py的MNIST Normalize。\n"
        " * 更新模型后重新运行tools/generate_camera_model_input.py，不要手工改scale。 */\n"
        "#ifndef CAMERA_MODEL_INPUT_H\n#define CAMERA_MODEL_INPUT_H\n"
        f"#define CAMERA_MODEL_INPUT_SCALE {float(scale)!r}\n"
        "#define CAMERA_MODEL_INPUT_ZERO_POINT 0\n"
        "#define CAMERA_MODEL_MEAN 0.1307f\n#define CAMERA_MODEL_STD 0.3081f\n#endif\n")
    print(args.output)


if __name__ == "__main__":
    main()
