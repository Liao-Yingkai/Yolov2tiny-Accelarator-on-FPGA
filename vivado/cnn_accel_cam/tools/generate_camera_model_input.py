#!/usr/bin/env python3
"""从训练导出的metadata生成少量标量；无依赖，无像素LUT，不修改训练工程。"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def quantization_parameters(scale):
    """只在离线计算参考；穷举通过才允许生成纯整数参数。"""
    mean, std = float32(0.1307), float32(0.3081)
    reference = []
    for gray in range(256):
        normalized = float32(float32(float32(gray / 255.0) - mean) / std)
        scaled = normalized / scale
        # 极小scale可能使double溢出；先饱和，无需对无穷值round。
        reference.append(-128 if scaled <= -128 else
                         127 if scaled >= 127 else round(scaled))
    coefficient = (1.0 / 255.0 / std) / scale
    offset = (-mean / std) / scale
    for shift in [20, *range(21, 31), *range(19, 0, -1)]:
        multiplier_scaled, offset_scaled = coefficient * (1 << shift), offset * (1 << shift)
        if not math.isfinite(multiplier_scaled) or not math.isfinite(offset_scaled):
            continue
        multiplier, bias = round(multiplier_scaled), round(offset_scaled)
        # 独立乘积及加法两端均须安全，不依赖抵消后的范围。
        if not (0 <= multiplier <= 2147483647 and
                -2147483648 <= bias <= 2147483647 and
                255 * multiplier <= 2147483647 and
                -2147483648 <= 255 * multiplier + bias <= 2147483647):
            continue
        actual = []
        for gray in range(256):
            value = gray * multiplier + bias
            integer, remainder = divmod(abs(value), 1 << shift)
            half = 1 << (shift - 1)
            integer += remainder > half or (remainder == half and integer % 2 != 0)
            actual.append(max(-128, min(127, -integer if value < 0 else integer)))
        if actual == reference:
            return multiplier, bias, shift, mean, std
    raise ValueError("找不到32位安全且全部256灰度严格等价的整数参数；未更新头文件")


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
    multiplier, bias, shift, mean, std = quantization_parameters(scale)
    args.output.write_text(
        "/* 自动生成；来源: " + str(args.metadata.resolve()) + "\n"
        f" * 离线参考: scale={float(scale)!r}, FP32 mean={mean!r}, std={std!r}。\n"
        " * 与原分步FP32归一化、double除scale、nearest-even及饱和对全部256灰度一致。\n"
        " * 更新模型后重新运行tools/generate_camera_model_input.py，不要手工改参数。 */\n"
        "#ifndef CAMERA_MODEL_INPUT_H\n#define CAMERA_MODEL_INPUT_H\n"
        "#define CAMERA_MODEL_INPUT_ZERO_POINT 0\n"
        f"#define CAMERA_MODEL_INPUT_MULTIPLIER {multiplier}\n"
        f"#define CAMERA_MODEL_INPUT_OFFSET ({bias})\n"
        f"#define CAMERA_MODEL_INPUT_SHIFT {shift}\n#endif\n")
    print(f"{args.output}: M={multiplier}, OFFSET={bias}, SHIFT={shift}; 256/256严格一致")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print(f"生成失败: {error}", file=sys.stderr)
        sys.exit(1)
