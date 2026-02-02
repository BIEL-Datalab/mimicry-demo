from __future__ import annotations
"""
生成 “Box_IN / Box_OUT / Box_NG” 坐标配置 JSON

用法示例：
  python generate_boxes_config.py --input-path frame_images/pad --output-path config/boxes_config/pad_boxes_config.json
  python generate_boxes_config.py --input-path frame_images/phone --output-path config/boxes_config/phone_boxes_config.json

输入目录需包含：
  - classes.txt
  - 一张图像文件（.jpg/.png/.jpeg），以及同名的 YOLO 标注 .txt
"""
import cv2
import json
import os
import numpy as np
import argparse
import sys
from typing import Optional, Dict, List, Tuple, TypeAlias
__all__ = [
    "read_image_safe",
    "get_boxes_from_yolo_txt",
    "resolve_input_paths",
    "output_dir_writable",
    "main",
]

def read_image_safe(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error reading image {path}: {e}")
        return None

def get_boxes_from_yolo_txt(image_path, txt_path, classes_path):
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes: List[str] = [line.strip() for line in f.readlines() if line.strip()]
    img = read_image_safe(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None
    height, width = img.shape[:2]
    print(f"Image resolution: {width}x{height}")
    BoxIntList: TypeAlias = List[int]
    BoxesConfig: TypeAlias = Dict[str, BoxIntList]
    boxes_config: BoxesConfig = {}
    target_classes: List[str] = ["Box_IN", "Box_OUT", "Box_NG"]
    if not os.path.exists(txt_path):
        print(f"Annotation file not found: {txt_path}")
        return {}

    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_idx = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            if cls_idx < 0 or cls_idx >= len(classes):
                continue
            class_name = classes[cls_idx]
            if class_name in target_classes:
                x1 = int((x_center - w / 2) * width)
                y1 = int((y_center - h / 2) * height)
                x2 = int((x_center + w / 2) * width)
                y2 = int((y_center + h / 2) * height)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)
                boxes_config[class_name] = [x1, y1, x2, y2]
    return boxes_config

def resolve_input_paths(input_dir: str) -> Optional[Tuple[str, str, str]]:
    input_dir = os.path.abspath(input_dir)
    classes_path = os.path.join(input_dir, "classes.txt")
    if not os.path.exists(classes_path):
        print(f"未找到 classes.txt: {classes_path}")
        return None
    candidates = []
    for name in os.listdir(input_dir):
        lower = name.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png"):
            stem = os.path.splitext(name)[0]
            txt = os.path.join(input_dir, stem + ".txt")
            if os.path.exists(txt):
                candidates.append((os.path.join(input_dir, name), txt))
    if not candidates:
        print(f"未找到成对的图像与标注文件: {input_dir}")
        return None
    image_path, txt_path = candidates[0]
    return image_path, txt_path, classes_path

def output_dir_writable(output_path: str) -> bool:
    abs_out = os.path.abspath(output_path)
    parent = os.path.dirname(abs_out) or "."
    parent = os.path.abspath(parent)
    if not os.path.exists(parent):
        print(f"输出目录不存在: {parent}")
        return False
    if not os.access(parent, os.W_OK):
        print(f"输出目录不可写: {parent}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="生成 Box_IN/Box_OUT/Box_NG 坐标配置 JSON")
    parser.add_argument("--input-path", default=".", help="输入目录，需包含 classes.txt、图像及同名 YOLO 标注 .txt")
    parser.add_argument("--output-path", default="boxes_config.json", help="输出 JSON 文件路径")
    args = parser.parse_args()
    input_dir = args.input_path
    output_path = args.output_path
    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        print(f"输入路径不存在或不是目录: {os.path.abspath(input_dir)}")
        sys.exit(1)
    resolved = resolve_input_paths(input_dir)
    if not resolved:
        sys.exit(1)
    image_path, txt_path, classes_path = resolved
    print(f"读取图像: {os.path.abspath(image_path)}")
    print(f"读取标注: {os.path.abspath(txt_path)}")
    print(f"读取类别: {os.path.abspath(classes_path)}")
    if not output_dir_writable(output_path):
        sys.exit(1)
    boxes = get_boxes_from_yolo_txt(image_path, txt_path, classes_path)
    if boxes:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(boxes, f, indent=4)
        print(f"已写出: {os.path.abspath(output_path)}")
        print(json.dumps(boxes, indent=4))
    else:
        print("未生成任何框配置")
        sys.exit(1)

if __name__ == "__main__":
    main()
