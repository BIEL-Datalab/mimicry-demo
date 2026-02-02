"""
视频首帧提取工具

功能：
1. 遍历输入目录的视频文件并过滤常见格式
2. 提取每个视频的第一帧并按指定格式无损保存
3. 根据文件名分隔规则生成子文件夹名称并创建目录
4. 将首帧图像保存到输出根目录下的对应子目录
5. 所有参数可通过命令行或配置文件设置

示例命令：
  python video_frame_extractor.py ^
    --input-dir d:/3_CODES/mimicry-demo/videos ^
    --output-root d:/3_CODES/mimicry-demo/data ^
    --delimiters "-_#" ^
    --rule range --start-index 3 --end-index 6 ^
    --image-format png --overwrite false
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Set, TypeAlias, Dict, Any

import cv2
import numpy as np


ImageArray: TypeAlias = np.ndarray
SUPPORTED_EXTS: Set[str] = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class ExtractorConfig:
    """配置对象
    参数：
      input_dir: 输入视频目录
      output_root: 输出根目录（在其下创建子目录）
      delimiters: 文件名分隔符集合字符串，例如 "-_#"
      rule: 子目录命名规则，"range" 或 "prefix"
      start_index: 当 rule=range 时，字段起始索引（1-based，包含）
      end_index: 当 rule=range 时，字段结束索引（1-based，包含）
      prefix_delimiter: 当 rule=prefix 时，用于取前缀的分隔符，默认使用第一个分隔符
      image_format: 保存格式 "png" 或 "jpg"
      overwrite: 是否覆盖已存在文件
      log_level: 日志级别
    返回：
      无
    """
    input_dir: Path
    output_root: Path
    delimiters: str = "-_#"
    rule: str = "range"
    start_index: int = 3
    end_index: int = 6
    prefix_delimiter: Optional[str] = None
    image_format: str = "png"
    overwrite: bool = False
    log_level: str = "INFO"

__all__ = [
    "ExtractorConfig",
    "load_config_file",
    "build_config",
    "configure_logging",
    "list_video_files",
    "split_filename_fields",
    "generate_subfolder_name",
    "extract_first_frame",
    "ensure_directory",
    "save_image",
    "process_videos",
    "parse_args",
    "main",
]

def load_config_file(path: Optional[Path]) -> Dict[str, Any]:
    """读取配置文件
    参数：
      path: 配置文件路径（JSON），可为 None
    返回：
      字典配置，若不存在或读取失败返回 {}
    """
    if not path:
        return {}
    try:
        text = path.read_text(encoding="utf-8")
        data: Dict[str, Any] = json.loads(text)
        return data
    except Exception as e:
        logging.warning(f"读取配置文件失败: {path} - {e}")
        return {}


def build_config(args: argparse.Namespace, from_file: dict) -> ExtractorConfig:
    """合并命令行与文件配置并构造配置对象
    参数：
      args: 命令行参数对象
      from_file: 来自配置文件的字典
    返回：
      ExtractorConfig 配置对象
    """
    def get(k: str, default):
        return getattr(args, k) if getattr(args, k) is not None else from_file.get(k, default)

    cfg = ExtractorConfig(
        input_dir=Path(get("input_dir", ".")),
        output_root=Path(get("output_root", "./data")),
        delimiters=get("delimiters", "-_#"),
        rule=get("rule", "range"),
        start_index=int(get("start_index", 3)),
        end_index=int(get("end_index", 6)),
        prefix_delimiter=get("prefix_delimiter", None),
        image_format=get("image_format", "png").lower(),
        overwrite=bool(get("overwrite", False)),
        log_level=get("log_level", "INFO").upper(),
    )
    return cfg


def configure_logging(level: str) -> None:
    """配置日志输出
    参数：
      level: 日志级别字符串，例如 "INFO"
    返回：
      None
    """
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def list_video_files(directory: Path, exts: Iterable[str] = SUPPORTED_EXTS) -> List[Path]:
    """遍历目录并过滤视频文件
    参数：
      directory: 输入目录 Path
      exts: 支持的扩展名集合（小写，含点）
    返回：
      视频文件路径列表
    """
    files: List[Path] = []
    if not directory.exists() or not directory.is_dir():
        logging.error(f"输入目录不存在或不是目录: {directory.resolve()}")
        return files
    allowed = {e.lower() for e in exts}
    for p in directory.iterdir():
        if p.is_file() and p.suffix.lower() in allowed:
            files.append(p)
    return files


def split_filename_fields(name: str, delimiters: str) -> List[str]:
    """按多个分隔符拆分文件名（不含扩展名）
    参数：
      name: 基础文件名（不含扩展名）
      delimiters: 分隔符集合字符串，如 "-_#"
    返回：
      拆分得到的字段列表
    """
    fields = [name]
    for d in delimiters:
        tmp: List[str] = []
        for f in fields:
            tmp.extend([x for x in f.split(d) if x != ""])
        fields = tmp
    return fields


def generate_subfolder_name(filename: str, delimiters: str, rule: str,
                            start_index: int, end_index: int,
                            prefix_delimiter: Optional[str]) -> str:
    """根据规则生成子目录名称
    参数：
      filename: 原始文件名（不含扩展名）
      delimiters: 分隔符集合字符串
      rule: 规则 "range" 或 "prefix"
      start_index: 字段起始索引（1-based，包含）- 仅在 range 使用
      end_index: 字段结束索引（1-based，包含）- 仅在 range 使用
      prefix_delimiter: 前缀分隔符 - 仅在 prefix 使用
    返回：
      子目录名称字符串（若为空则返回 "misc"）
    """
    if rule == "prefix":
        sep = prefix_delimiter or (delimiters[0] if delimiters else "-")
        parts = filename.split(sep)
        sub = parts[0] if parts else filename
    else:
        fields = split_filename_fields(filename, delimiters)
        s = max(1, start_index)
        e = max(s, end_index)
        sub = "_".join(fields[s - 1:e]) if fields else filename
    sub = sub.strip()
    return sub if sub else "misc"


def extract_first_frame(video_path: Path) -> Optional[np.ndarray]:
    """提取视频的第一帧
    参数：
      video_path: 视频文件路径
    返回：
      成功返回图像 ndarray（BGR），失败返回 None
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"无法打开视频: {video_path.resolve()}")
            return None
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            logging.error(f"读取第一帧失败: {video_path.resolve()}")
            return None
        return frame
    except Exception as e:
        logging.exception(f"提取第一帧异常: {video_path.resolve()} - {e}")
        return None


def ensure_directory(path: Path) -> None:
    """确保目录存在
    参数：
      path: 需要创建的目录路径
    返回：
      None
    """
    path.mkdir(parents=True, exist_ok=True)


def save_image(image: np.ndarray, output_dir: Path, original_name: str,
               image_format: str = "png", overwrite: bool = False) -> Path:
    """保存图像到目标目录
    参数：
      image: 图像数据（BGR）
      output_dir: 目标目录
      original_name: 原始视频文件名（包含扩展名）
      image_format: 保存格式 "png"/"jpg"
      overwrite: 是否覆盖已存在文件
    返回：
      保存后的文件路径
    """
    ensure_directory(output_dir)
    stem = Path(original_name).stem
    suffix = ".png" if image_format.lower() == "png" else ".jpg"
    out_path = output_dir / f"{stem}{suffix}"
    if out_path.exists() and not overwrite:
        logging.info(f"已存在，跳过: {out_path.resolve()}")
        return out_path
    params: List[int] = []
    if suffix == ".jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, 100]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    ok, buf = cv2.imencode(suffix, image, params)
    if not ok or buf is None:
        raise RuntimeError(f"编码失败: {out_path.resolve()}")
    with out_path.open("wb") as f:
        f.write(buf.tobytes())
    logging.info(f"已保存首帧: {out_path.resolve()}")
    return out_path


def process_videos(cfg: ExtractorConfig) -> None:
    """处理所有视频文件
    参数：
      cfg: 配置对象
    返回：
      None
    """
    configure_logging(cfg.log_level)
    input_dir = cfg.input_dir.resolve()
    output_root = cfg.output_root.resolve()
    logging.info(f"输入目录: {input_dir}")
    logging.info(f"输出根目录: {output_root}")
    videos = list_video_files(input_dir, SUPPORTED_EXTS)
    if not videos:
        logging.warning("未发现可处理的视频文件")
        return
    for vp in videos:
        try:
            filename_no_ext = vp.stem
            subfolder_name = generate_subfolder_name(
                filename_no_ext,
                cfg.delimiters,
                cfg.rule,
                cfg.start_index,
                cfg.end_index,
                cfg.prefix_delimiter,
            )
            target_dir = output_root / subfolder_name
            frame = extract_first_frame(vp)
            if frame is None:
                logging.error(f"处理失败（首帧为空）: {vp.resolve()}")
                continue
            save_image(frame, target_dir, vp.name, cfg.image_format, cfg.overwrite)
            logging.info(f"处理完成: {vp.name} -> {target_dir.name}")
        except Exception as e:
            logging.exception(f"处理视频失败: {vp.resolve()} - {e}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数
    参数：
      argv: 参数列表，可为 None 使用 sys.argv
    返回：
      argparse.Namespace
    """
    p = argparse.ArgumentParser(description="视频首帧提取工具")
    p.add_argument("--config", type=str, default="config/video_extractor_config/extractor_config.json", help="配置文件路径（JSON），命令行优先于配置文件")
    p.add_argument("--input-dir", type=str, default=None, help="输入视频目录")
    p.add_argument("--output-root", type=str, default=None, help="输出根目录（在此目录下创建子目录）")
    p.add_argument("--delimiters", type=str, default=None, help="文件名分隔符集合字符串，例如 \"-_#\"")
    p.add_argument("--rule", type=str, default=None, choices=["range", "prefix"], help="子目录命名规则")
    p.add_argument("--start-index", type=int, default=None, help="字段起始索引（1-based，包含），仅在 rule=range 时使用")
    p.add_argument("--end-index", type=int, default=None, help="字段结束索引（1-based，包含），仅在 rule=range 时使用")
    p.add_argument("--prefix-delimiter", type=str, default=None, help="rule=prefix 时的前缀分隔符")
    p.add_argument("--image-format", type=str, default=None, choices=["png", "jpg"], help="保存格式")
    p.add_argument("--overwrite", type=str, default=None, help="是否覆盖已存在文件（true/false）")
    p.add_argument("--log-level", type=str, default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="日志级别")
    args = p.parse_args(argv)
    if args.overwrite is not None:
        val = str(args.overwrite).lower()
        args.overwrite = val in {"1", "true", "yes", "y"}
    return args


def main() -> None:
    """主入口
    参数：
      无
    返回：
      None
    """
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    default_cfg = script_dir / "config" / "video_extractor_config" / "extractor_config.json"
    cfg_path = Path(args.config) if args.config else default_cfg
    cfg_dict = load_config_file(cfg_path)
    cfg = build_config(args, cfg_dict)
    process_videos(cfg)


if __name__ == "__main__":
    main()
