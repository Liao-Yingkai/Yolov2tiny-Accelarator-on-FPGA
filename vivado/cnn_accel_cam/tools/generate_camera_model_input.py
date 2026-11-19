#!/usr/bin/env python3
"""从YOLO metadata生成整数输入量化标量；无依赖、无LUT、不修改训练工程。"""
import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path


def float32(value):
    """显式模拟每步IEEE FP32舍入；仅离线使用。"""
    return struct.unpack("f", struct.pack("f", value))[0]


def round_shift_even(value, shift):
    integer, remainder = divmod(value, 1 << shift)
    half = 1 << (shift - 1)
    return integer + (remainder > half or (remainder == half and integer % 2))


def quantization_parameters(scale):
    """参考to_tensor的byte/255及torch FP32 reciprocal，证明全部256输入。"""
    scale32 = float32(scale)
    if not math.isfinite(scale32) or scale32 <= 0:
        raise ValueError("input_scale转FP32后必须仍为有限正数")
    inverse = float32(1.0 / scale32)
    if not math.isfinite(inverse):
        raise ValueError("FP32 reciprocal必须有限")
    reference = [max(-128, min(127, round(float32(float32(pixel / 255.0) * inverse))))
                 for pixel in range(256)]
    # 每个目标值给出M的精确整数区间，包含nearest-even的开/闭半整数边界。
    # 先尝试Q20，再尝试其余S；不靠局部猜测而漏掉可行的32位安全解。
    for shift in [20, *range(21, 31), *range(19, 0, -1)]:
        lower, upper = 0, 2147483647 // 255
        denominator = 1 << shift
        for pixel, target in enumerate(reference[1:], 1):
            divisor = 2 * pixel
            if target > 0:
                bound = (2 * target - 1) * denominator
                lower = max(lower, (bound + divisor - 1) // divisor if target % 2 == 0
                            else bound // divisor + 1)
            if target < 127:
                bound = (2 * target + 1) * denominator
                upper = min(upper, bound // divisor if target % 2 == 0
                            else (bound - 1) // divisor)
        if lower > upper:
            continue
        multiplier = max(lower, min(upper, round(inverse / 255.0 * denominator)))
        actual = [min(127, round_shift_even(pixel * multiplier, shift))
                  for pixel in range(256)]
        if actual != reference:
            raise ValueError("整数区间证明失败；未更新头文件")
        return multiplier, shift
    raise ValueError("找不到32位安全且全部256字节严格等价的整数参数；未更新头文件")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path,
                        help="metadata.json路径；未指定时使用YOLO_DATA_DIR/metadata.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "vitis/cnn_accel_cam/src/camera_model_input.h")
    args = parser.parse_args()
    if args.metadata is None:
        data_dir = os.environ.get("YOLO_DATA_DIR")
        if not data_dir:
            parser.error("请指定--metadata，或设置YOLO_DATA_DIR为包含metadata.json的导出目录")
        args.metadata = Path(data_dir) / "metadata.json"
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 2 or metadata.get("weight_layout_version") != 3:
        raise ValueError("要求metadata format_version=2、weight_layout_version=3")
    first = metadata["layers"][0]
    config = metadata["model_config"]
    if (first["name"] != "conv1" or first["original_oihw_shape"] != [16, 3, 3, 3]
            or first["Cin"] != 3 or first["Cout"] != 16
            or config["input_height"] != 416 or config["input_width"] != 416):
        raise ValueError("要求416×416 RGB、conv1 OIHW=[16,3,3,3]")
    scales, zeros = first["input_scale"], first["input_zero_point"]
    if not isinstance(scales, list) or len(scales) != 1:
        raise ValueError("conv1.input_scale必须是单元素数组")
    scale = scales[0]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("conv1.input_scale必须为有限正数")
    if (not isinstance(zeros, list) or len(zeros) != 1 or isinstance(zeros[0], bool)
            or not isinstance(zeros[0], (int, float)) or zeros[0] != 0):
        raise ValueError("仅支持conv1.input_zero_point=[0]")
    multiplier, shift = quantization_parameters(scale)
    # 所有读取、契约检查及256值证明先完成，失败时不覆盖现有头文件。
    args.output.write_text(
        "/* 自动生成；来源: " + args.metadata.name + "（与模型权重同批导出）\n"
        f" * conv1 input_scale={scale!r}; metadata v2 / weight layout v3。\n"
        f" * checkpoint SHA256: {metadata.get('checkpoint', {}).get('sha256', 'unknown')}\n"
        " * FP32 byte/255、FP32 reciprocal乘法、nearest-even及INT8饱和：256/256一致。\n"
        " * 更新模型后重新运行tools/generate_camera_model_input.py，不要手改参数。 */\n"
        "#ifndef CAMERA_MODEL_INPUT_H\n#define CAMERA_MODEL_INPUT_H\n"
        "#define CAMERA_MODEL_INPUT_ZERO_POINT 0\n"
        f"#define CAMERA_MODEL_INPUT_MULTIPLIER {multiplier}\n"
        f"#define CAMERA_MODEL_INPUT_SHIFT {shift}\n#endif\n", encoding="utf-8")
    print(f"{args.output}: M={multiplier}, SHIFT={shift}; 256/256严格一致")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        print(f"生成失败: {error}", file=sys.stderr)
        sys.exit(1)
