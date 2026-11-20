#!/usr/bin/env python3
"""从YOLO metadata离线生成输出解码常量；仅依赖Python标准库。"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path


def positive_float32(value):
    """拒绝布尔值、非有限值和FP32上溢/下溢。"""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError("量化尺度及anchor必须为有限正数")
    rounded = struct.unpack("f", struct.pack("f", value))[0]
    if not math.isfinite(rounded) or rounded <= 0:
        raise ValueError("转FP32后必须仍为有限正数")
    return rounded


def c_float(value):
    # 九位有效数字足以往返FP32；整数形式补小数点，避免非法的1f。
    literal = format(positive_float32(value), ".9g")
    if "." not in literal and "e" not in literal:
        literal += ".0"
    return literal + "f"


def generate(metadata, source):
    if metadata.get("format_version") != 2 or metadata.get("weight_layout_version") != 3:
        raise ValueError("要求metadata format_version=2、weight_layout_version=3")
    config = metadata["model_config"]
    if (config["input_height"] != 416 or config["input_width"] != 416
            or config["num_classes"] != 20 or config["num_anchors"] != 5):
        raise ValueError("要求416×416、20类、5个anchor")
    last = metadata["layers"][-1]
    if (last["name"] != "prediction" or last["kernel"] != [1, 1]
            or last["Cin"] != 1024 or last["Cout"] != 125
            or last["original_oihw_shape"] != [125, 1024, 1, 1]
            or last["logical_output_channels"] != 125
            or last["physical_output_channels"] != 128):
        raise ValueError("要求末层prediction 1×1、Cin=1024、Cout=125、物理通道128")
    scales, zeros = last["output_scale"], last["output_zero_point"]
    if not isinstance(scales, list) or len(scales) != 1:
        raise ValueError("prediction.output_scale必须是单元素数组")
    scale_literal = c_float(scales[0])
    if (not isinstance(zeros, list) or len(zeros) != 1
            or isinstance(zeros[0], bool) or not isinstance(zeros[0], (int, float))
            or zeros[0] != 0):
        raise ValueError("仅支持prediction.output_zero_point=[0]")
    anchors = config["anchors"]
    if (not isinstance(anchors, list) or len(anchors) != 5
            or any(not isinstance(pair, list) or len(pair) != 2 for pair in anchors)):
        raise ValueError("anchors必须为5×2数组，保持metadata顺序")
    anchor_literal = "{ " + ", ".join(
        "{ " + ", ".join(c_float(v) for v in pair) + " }" for pair in anchors) + " }"
    names = config["classes"]
    if (not isinstance(names, list) or len(names) != 20
            or any(not isinstance(name, str) or not name or not name.isascii()
                   or len(name) > 16 or any(ord(c) < 32 or ord(c) == 127 for c in name)
                   for name in names)):
        raise ValueError("classes必须为20个非空可打印ASCII名称，每项不超过16字节")
    name_literal = "{ " + ", ".join(json.dumps(name, ensure_ascii=True) for name in names) + " }"
    # 只输出宏；消费模块自行定义一次静态表，不在每个包含单元复制数据。
    return (
        "/* 自动生成；来源: " + Path(source).name + "（与模型权重同批导出）\n"
        f" * checkpoint SHA256: {metadata.get('checkpoint', {}).get('sha256', 'unknown')}\n"
        f" * prediction output_scale={scales[0]!r}; FP32={scale_literal}。\n"
        " * 更新模型后重新运行tools/generate_camera_model_output.py；勿手改。 */\n"
        "#ifndef CAMERA_MODEL_OUTPUT_H\n#define CAMERA_MODEL_OUTPUT_H\n"
        f"#define CAMERA_MODEL_OUTPUT_SCALE {scale_literal}\n"
        "#define CAMERA_MODEL_OUTPUT_ZERO_POINT 0\n"
        "#define CAMERA_MODEL_OUTPUT_GRID 13\n"
        "#define CAMERA_MODEL_OUTPUT_LOGICAL_CHANNELS 125\n"
        "#define CAMERA_MODEL_OUTPUT_PHYSICAL_CHANNELS 128\n"
        "#define CAMERA_MODEL_OUTPUT_CLASSES 20\n"
        "#define CAMERA_MODEL_OUTPUT_ANCHORS 5\n"
        f"#define CAMERA_MODEL_OUTPUT_ANCHOR_VALUES {anchor_literal}\n"
        f"#define CAMERA_MODEL_OUTPUT_CLASS_NAMES {name_literal}\n"
        "#endif\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path,
                        default=Path(__file__).resolve().parents[3] / "data/yolov2_tiny/metadata.json",
                        help="metadata.json路径；默认仓库data/yolov2_tiny/metadata.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "vitis/cnn_accel_cam/src/camera_model_output.h")
    args = parser.parse_args()
    if not args.metadata.is_file():
        parser.error(f"找不到 {args.metadata}；请按 hls/docs/data.md 准备数据，或指定 --metadata")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    # 完整校验与格式化成功后才替换头文件；错误metadata不破坏已有产物。
    header = generate(metadata, args.metadata.resolve())
    args.output.write_text(header, encoding="utf-8")
    print(f"{args.output}: prediction尺度、5个anchor及20类名称已生成")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as error:
        print(f"生成失败: {error}", file=sys.stderr)
        sys.exit(1)
