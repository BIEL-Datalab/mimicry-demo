# mimicry-demo

基于 YOLOv8 的 FQC 工序“动态识别系统”示例项目。系统通过识别员工手部与物料筐的交互，自动判定一次检查任务的开始与结束，统计产品在检查过程中的横向/纵向时长，区分平板玻璃与手机玻璃，并将带有 AI 标注叠加的完整过程保存为 MP4。

**Demo片段展示**

![功能演示](assets/demo_segment.gif)


**项目结构**
```
mimicry-demo/
├── README.md                            # 项目说明文档
├── config/                              # 项目配置目录
│   ├── boxes_config/                    # 三框静态配置
│   │   ├── pad_boxes_config.json        # Pad 工位三框配置
│   │   └── phone_boxes_config.json      # Phone 工位三框配置
│   ├── data_config/                     # 数据集 YAML 配置
│   │   └── mimicry_20260121_data.yaml   # 类别与路径定义
│   └── video_extractor_config/          # 视频提取工具配置
│       └── extractor_config.json        # 工具默认配置
├── frame_images/                        # Boxes 标注与样例数据
│   ├── pad/                             # Pad 工位样例
│   │   ├── classes.txt                  # 类别清单
│   │   ├── pad_XXX.jpg                  # 首帧图像
│   │   └── pad_XXX.txt                  # YOLO 标注
│   └── phone/                           # Phone 工位样例
│       ├── classes.txt                  # 类别清单
│       ├── phone_XXX.jpg                # 首帧图像
│       └── phone_XXX.txt                # YOLO 标注
├── demo_pad_glass.py                    # Pad 工位推理与业务逻辑
├── demo_phone_glass.py                  # Phone 工位推理与业务逻辑
├── generate_boxes_config.py             # 从标注生成三框配置
├── requirements.txt                     # 依赖清单
├── video_frame_extractor.py             # 视频首帧提取工具
└── videos/                              # 推理视频输入目录
    └── raw/                             # 原始输入视频
        ├── pad-glass-XXX.avi            # Pad 工位示例视频
        └── phone-glass-XXX.avi          # Phone 工位示例视频
```

**当前架构与功能说明**
- 自动识别任务周期的开始/结束（统一采用 Hand 与 Box_IN 的 IoM 阈值触发）
- 统计周期内的横向/纵向时长（宽高比判定）
- 智能类别投票（周期内按帧对 Pad/Phone 计票，周期结束取最高票）
- 叠加显示累计 OK/NG 计数
- 输出带标注的完整 MP4 与周期开始/结束快照

**环境与安装**
- 操作系统：Windows / Linux / macOS
- Python：3.9–3.11
- 依赖：ultralytics（YOLOv8）、opencv-python（4.x）、numpy、pyyaml、torch（与本机 CUDA/CPU 匹配）

```bash
# 创建并启用虚拟环境（可选）
python -m venv .venv
.\\.venv\\Scripts\\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -U pip
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
python -c "import cv2, ultralytics; print('OK', cv2.__version__)"
```

**类别与数据集**
- 模型类别顺序需与数据集一致：`["Hand", "Pad_Glass", "Phone_Glass"]`
- 数据集 YAML 示例（请按本地路径设置）：`d:/3_CODES/_YOLO/YOLO/dataset/mimicry_20260121_data.yaml`
- 模型权重示例（请按训练输出设置）：`runs/train/<你的实验>/weights/best.pt`

**静态区域配置（必须）**
三框坐标 JSON 已提供示例：
- Pad 工位：`config/boxes_config/pad_boxes_config.json`
- Phone 工位：`config/boxes_config/phone_boxes_config.json`

```json
{
  "Box_IN":  [x1, y1, x2, y2],
  "Box_OUT": [x1, y1, x2, y2],
  "Box_NG":  [x1, y1, x2, y2]
}
```

**视频帧提取工具使用**
- 配置文件：`config/video_extractor_config/extractor_config.json`
- 运行示例（使用默认配置）：
```bash
python video_frame_extractor.py --config config/video_extractor_config/extractor_config.json
```


**模块依赖关系**
- demo_pad_glass.py / demo_phone_glass.py 依赖：ultralytics.YOLO、opencv-python、numpy、pyyaml、json
- video_frame_extractor.py 依赖：opencv-python、numpy、argparse、pathlib、json、logging
- generate_boxes_config.py 依赖：opencv-python、numpy、argparse、json、os、sys

**操作指南**
- Pad 工位运行（需自行设置 model_path 与数据集 YAML 路径）：
```python
from demo_pad_glass import process_video_stream
process_video_stream(
    video_path="videos/pad-glass-BA远程13_L3-3F-313#照检查台_20260108085800_20260108110000.avi",
    model_path="runs/train/<你的实验>/weights/best.pt",
    static_boxes_path="config/boxes_config/pad_boxes_config.json",
)
```
- Phone 工位运行：
```python
from demo_phone_glass import process_video_stream
process_video_stream(
    video_path="videos/phone-glass-BA远程13_L3-3F-318#照检查台_20260103075300_20260103103000.avi",
    model_path="runs/train/<你的实验>/weights/best.pt",
    static_boxes_path="config/boxes_config/phone_boxes_config.json",
)
```
- 提取首帧到子目录（使用默认配置文件）：
```bash
python video_frame_extractor.py --config config/video_extractor_config/extractor_config.json
```

**业务逻辑摘要**
- 周期开始：Hand 与 Box_IN 的 IoM > hand_glass_threshold（默认 0.6）
- 周期结束：
  - Pad：离开过 Box_IN 后再次进入 Box_IN 判定 OK；进入 Box_NG 判定 NG；最短时长 ≥ fps*3 秒
  - Phone：进入 Box_OUT 或 Box_NG，IoM 较大者决定 OK/NG；最短时长 ≥ fps*3 秒
- 姿态统计：box_ratio=0.9 判定横/纵并累计时长
- 叠加层：Pad 显示累计 OK/NG；Phone 显示累计 OK/NG 与 LAST

**输出结果**
- 完整过程视频：`demo_results/<输入文件名>_result.mp4`
- 关键帧快照：`shortcut/<glass_count>_Begin.png`、`shortcut/<glass_count>_END.png`
- 控制台日志：周期 ID、时间戳、检测结果、总时长、横纵向时长

**关键参数**
- hand_glass_threshold：IoM 阈值（默认 0.6）
- box_ratio：横/纵判断的宽高比阈值（默认 0.9）
- MIN_CYCLE_FRAMES：最短周期帧数（默认 fps*3.0）

**常见问题**
- 周期无法开始/结束：检查三框坐标与阈值设置是否合理
- 类别名不匹配：确认数据集 YAML 的 `names` 顺序与模型训练一致
- 模型无法加载：检查 `model_path` 路径与权限，确认 PyTorch/Ultralytics 安装正确
- 输出不清晰：项目使用原始分辨率写出；确保输入视频质量与帧率适配

