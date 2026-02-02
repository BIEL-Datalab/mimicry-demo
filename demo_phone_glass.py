import cv2
import numpy as np
import time
import math
import threading
import itertools
import uuid
from collections import deque, Counter
from collections import defaultdict
import os
import re
import json
from ultralytics import YOLO

def load_dataset_names(yaml_path: str) -> list[str]:
    """
    加载数据集 YAML 文件中的类别名称
    参数:
      yaml_path: YAML 文件路径
    返回:
      类别名称列表
    """
    names = []
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            names = data.get("names", [])
    except Exception:
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                content = f.read()
            m = re.search(r"names:\s*\[(.*?)\]", content, re.S)
            if m:
                raw = "[" + m.group(1) + "]"
                names = eval(raw)
        except Exception:
            names = []
    return names


class ObjectTracker:
    """
    业务对象跟踪器（手机玻璃工位）
    职责:
      - 管理每帧检测与周期状态
      - 判定周期起止与 OK/NG 结果
      - 统计姿态时长并输出历史记录
    """
    def __init__(self, detector: YOLO, dataset_names: list[str], static_boxes_path: str = "sg_boxes_config.json", fps: float = 30, hand_glass_threshold: float = 0.6) -> None:
        """
        初始化跟踪器
        参数:
          detector: YOLO 检测器实例
          dataset_names: 类别名称列表
          static_boxes_path: 静态框配置路径
          fps: 视频帧率
          hand_glass_threshold: 手与框的交并比阈值
        """
        self.fps = fps  # 新增帧率属性
        self.frame_count = 0  # 新增帧计数器
        self.MIN_CYCLE_FRAMES = int(fps * 3.0)  # 转换为帧数阈值
        self.detector = detector
        self.dataset_names = dataset_names or []
        self.static_boxes_path = static_boxes_path
        self.hand_glass_threshold = hand_glass_threshold
        self.box_ratio = 0.9
        self.lock = threading.Lock()
        self.glass_count = 0
        self.current_box = None
        self.current_glass_type = None  # 当前周期检测到的玻璃类型
        self.cumulative_stats = {
            "Pad_Glass": {"OK": 0, "NG": 0, "LAST": None},
            "Phone_Glass": {"OK": 0, "NG": 0, "LAST": None}
        }

        self.class_list = ["Box_OUT", "Box_NG", "Box_IN", "Hand", "Pad_Glass", "Phone_Glass"]
        # 状态跟踪，Hand改为存储列表
        self.last_boxes = {
            "Box_OUT": None,
            "Box_NG": None,
            "Box_IN": None,
            "Hand": [],  # 存储多个手部检测框
            "Pad_Glass": None,
            "Phone_Glass": None,
        }
        self.period = False
        self.in_cycle = False
        self.cycle_start = None
        self.current_orientation = None
        self.orientation_timestamps = []
        self.history = deque(maxlen=5)  # 修改为固定长度队列
        self.has_left_start_box = False  # 标记是否离开起始区域
        self.cycle_done = True
        self.load_static_boxes()

    def load_static_boxes(self) -> None:
        """
        加载静态框配置并更新缓存
        """
        try:
            with open(self.static_boxes_path, "r") as f:
                static_boxes = json.load(f)
                for name, box in static_boxes.items():
                    if name in self.last_boxes:
                        self.last_boxes[name] = box
        except FileNotFoundError:
            print(f"{self.static_boxes_path} not found, skipping static box initialization")
        except Exception as e:
            print(f"Error loading {self.static_boxes_path}: {e}")

    @staticmethod
    def compute_iom(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
        """
        计算两框的最小并比（IoM）
        参数:
          box1, box2: 边界框坐标 (x0,y0,x1,y1)
        返回:
          交集面积 / 两者较小面积
        """
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        inter_width = max(0, xB - xA)
        inter_height = max(0, yB - yA)
        inter_area = inter_width * inter_height

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        min_area = min(area1, area2)

        if min_area == 0:
            return 0.0

        iom = inter_area / min_area
        return iom

    @staticmethod
    def calculate_iou(box1: tuple[float, float, float, float] | None, box2: tuple[float, float, float, float] | None) -> float:
        """
        计算两框的 IoU
        参数:
          box1, box2: 边界框坐标或 None
        返回:
          交并比数值（0~1）
        """
        if box1 is None or box2 is None:
            return 0.0

        x0, y0, x1, y1 = box1
        x2, y2, x3, y3 = box2

        if x0 > x1:
            x0, y0, x1, y1 = x1, y1, x0, y0
        if x2 > x3:
            x2, y2, x3, y3 = x3, y3, x2, y2

        s1 = (x1 - x0) * (y1 - y0)
        s2 = (x3 - x2) * (y3 - y2)
        w = max(0, min(x1, x3) - max(x0, x2))
        h = max(0, min(y1, y3) - max(y0, y2))
        inter = w * h
        iou = inter / (s1 + s2 - inter + 1e-6)
        return iou

    @staticmethod
    def calculate_center(box: tuple[float, float, float, float] | list | None) -> tuple[float, float] | None:
        """
        计算框中心点坐标
        参数:
          box: 边界框或 None/空列表
        返回:
          (cx, cy) 或 None
        """
        if box is None or (isinstance(box, list) and not box):
            return None
        x0, y0, x1, y1 = box
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    @staticmethod
    def point_in_box(point: tuple[float, float] | None, box: tuple[float, float, float, float] | None) -> bool:
        """
        点是否位于框内
        参数:
          point: 点坐标 (x,y)
          box: 边界框 (x0,y0,x1,y1)
        返回:
          True/False
        """
        if box is None or point is None:
            return False
        x, y = point
        x0, y0, x1, y1 = box
        return x0 <= x <= x1 and y0 <= y <= y1

    def update_boxes(self, current_detections: dict[str, list[tuple[tuple[float, float, float, float], float]]]) -> None:
        """
        更新动态检测框缓存（静态三框不被覆盖）
        参数:
          current_detections: {类别: [(bbox, conf), ...]}
        """
        for cls in self.class_list:
            # 如果是静态区域，则跳过更新，防止检测结果覆盖静态配置
            if cls in ["Box_IN", "Box_OUT", "Box_NG"]:
                continue

            detected_items = current_detections.get(cls)
            if detected_items:
                if cls == "Hand":  # 手的box保留前两个，其他的保留置信度度最大的一个
                    sorted_items = sorted(
                        detected_items, key=lambda x: x[1], reverse=True
                    )
                    self.last_boxes[cls] = [item[0] for item in sorted_items[:2]]
                else:
                    sorted_items = sorted(
                        detected_items, key=lambda x: x[1], reverse=True
                    )
                    self.last_boxes[cls] = sorted_items[0][0]
            elif cls in ["Hand", "Pad_Glass", "Phone_Glass"]:
                # 如果没有检测到手或玻璃，清空对应项，防止残留
                if cls == "Hand":
                    self.last_boxes[cls] = []
                else:
                    self.last_boxes[cls] = None

    def detect_orientation(self, glass_box: tuple[float, float, float, float] | None, threshold: float = 0.92) -> str | None:
        """
        根据玻璃框宽高比判断姿态方向
        参数:
          glass_box: 玻璃边界框
          threshold: 宽高比阈值（大于阈值为横向）
        返回:
          'horizontal' / 'vertical' / None
        """
        # 输入校验：空框直接返回 None
        if glass_box is None:
            return None
        w = glass_box[2] - glass_box[0]
        h = glass_box[3] - glass_box[1]
        ratio = w / h
        # 核心意图：以宽高比为依据判定横纵向
        if ratio > threshold:
            return "horizontal"
        else:
            return "vertical"

    def hand_process(self, hands: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
        """
        手部框筛选与排序
        参数:
          hands: 手部边界框列表
        返回:
          处理后的边界框（最多两个，按左到右）
        """
        handssum = len(hands)
        if handssum > 2:
            for hand_i in hands:
                if hand_i[2] < 250:
                    hands.remove(hand_i)

        if len(hands) > 2:
            hands_list = list(itertools.combinations(range(handssum), 2))
            for i, j in hands_list:
                if self.calculate_iou(hands[i], hands[j]) > 0.1:
                    hands = [hands[i], hands[j]]
                    break

        if len(hands) == 2:
            if hands[0][0] > hands[1][0]:
                hands[0], hands[1] = hands[1], hands[0]

        return hands

    def glass_process(self, glass: list[tuple[tuple[float, float, float, float], float]] | None) -> list[tuple[tuple[float, float, float, float], float]] | None:
        """
        玻璃框筛选
        参数:
          glass: [(bbox, conf), ...]
        返回:
          仅保留最高置信度的一个
        """
        # 只保留置信度最高的一个box，如果执行度一样则取第一个box
        if glass is not None and len(glass) > 1:
            glass = sorted(glass, key=lambda x: x[1], reverse=True)
            glass = [glass[0]]
        return glass

    def process_frame(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        处理单帧图像
        参数:
          img: 输入 BGR 帧
        返回:
          (标注后的图像, 周期结束标识)
        关键步骤:
          - 坐标缩放与检测归集
          - 选择当前参与逻辑的玻璃框
          - 判定周期开始与结束（Box_IN/OUT/NG）
          - 叠加可视化与统计信息
        """
        self.frame_count += 1  # 每帧递增计数器
        orig_h, orig_w = img.shape[:2]
        scale_x = orig_w / 640.0
        scale_y = orig_h / 480.0
        # 根据分辨率动态调整线宽和字体大小，以保持清晰度
        base_thickness = max(1, int(2 * scale_x))
        base_font_scale = 0.35 * scale_x

        with self.lock:
            img_resized = cv2.resize(img, (640, 480))
            results = self.detector(img_resized, verbose=False)
            current_detections = defaultdict(list)

            img_np = img.copy() # 使用原始高分辨率图像进行标注

            r = results[0]
            xyxy = r.boxes.xyxy
            confs = r.boxes.conf
            clss = r.boxes.cls
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].cpu().numpy().tolist()
                # 将坐标缩放回原始分辨率
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y

                conf = float(confs[i].item())
                cls_idx = int(clss[i].item())
                if 0 <= cls_idx < len(self.dataset_names):
                    raw_name = self.dataset_names[cls_idx]
                    current_detections[raw_name].append(((x1, y1, x2, y2), conf))

            self.update_boxes(current_detections)

            hand_boxes = self.last_boxes["Hand"]
            box_in = self.last_boxes["Box_IN"]
            box_out = self.last_boxes["Box_OUT"]
            box_ng = self.last_boxes["Box_NG"]
            pad_box = self.last_boxes["Pad_Glass"]
            phone_box = self.last_boxes["Phone_Glass"]

            sel_glass_box = None
            sel_glass_label = None
            pad_list = current_detections.get("Pad_Glass", [])
            phone_list = current_detections.get("Phone_Glass", [])
            if pad_list or phone_list:
                candidates = []
                for (b, c) in pad_list:
                    candidates.append(("Pad_Glass", b, c))
                for (b, c) in phone_list:
                    candidates.append(("Phone_Glass", b, c))
                candidates.sort(key=lambda x: x[2], reverse=True)
                sel_glass_label, sel_glass_box, _ = candidates[0]
            else:
                if phone_box is not None:
                    sel_glass_label, sel_glass_box = "Phone_Glass", phone_box
                elif pad_box is not None:
                    sel_glass_label, sel_glass_box = "Pad_Glass", pad_box
            glass_box = sel_glass_box
            if self.in_cycle and sel_glass_label is not None:
                self.current_glass_type = sel_glass_label
                self.glass_type_votes[sel_glass_label] += 1 # 记录类别投票
            # hand_boxes = self.hand_process(hand_boxes)
            # glass_box = self.glass_process(glass_box)

            if not self.in_cycle:
                for hand_box in hand_boxes:
                    if box_in is not None:
                        if not self.cycle_done:
                            print("Duplicate Box_IN ignored")
                            continue
                        iom = self.compute_iom(hand_box, box_in)
                        if iom > self.hand_glass_threshold:
                            print(f"Cycle Start miou: {iom}")
                            self.start_cycle()
                            cv2.imwrite(f"shortcut/{self.glass_count}_Begin.png", img_np)
                            break

            if self.in_cycle:
                end_cycle = False
                for hand_box in hand_boxes:
                    if (
                        box_out is not None
                        and self.compute_iom(hand_box, box_out)
                        > self.hand_glass_threshold
                    ) or (
                        box_ng is not None
                        and self.compute_iom(hand_box, box_ng)
                        > self.hand_glass_threshold
                    ):
                        self.current_box = (
                            "OK"
                            if self.compute_iom(hand_box, box_out)
                            > self.compute_iom(hand_box, box_ng)
                            else "NG"
                        )
                        end_cycle = True
                        break

                if end_cycle:
                    current_frames = self.frame_count - self.cycle_start_frame
                    if current_frames >= self.MIN_CYCLE_FRAMES:
                        print(
                            f"Cycle End at miou: {self.compute_iom(hand_box, box_out)}"
                        )
                        cv2.imwrite(f"shortcut/{self.glass_count}_END.png", img_np)
                        self.end_cycle()

            if self.in_cycle and glass_box is not None:
                self.process_orientation(glass_box)

            colors = {
                "Box_IN": (50, 205, 50),    # 亮绿色（入框区域）
                "Box_OUT": (71, 99, 255),   # 柔红/珊瑚（出框区域）
                "Box_NG": (255, 127, 80),   # 珊瑚/橙色（不良区域）
                "Hand": (0, 255, 255),      # 青色（手部）
                "Pad_Glass": (255, 0, 255), # 洋红（平板玻璃）
                "Phone_Glass": (255, 0, 200), # 粉紫（手机玻璃）
            }

            for cls_name, detections in current_detections.items():
                for box, conf in detections:
                    x1, y1, x2, y2 = map(int, box)
                    color = colors.get(cls_name, (200, 200, 200))
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, base_thickness)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, color, base_thickness)

            # 额外高亮当前参与逻辑处理的玻璃框
            if glass_box is not None and sel_glass_label is not None:
                x1, y1, x2, y2 = map(int, glass_box)
                color = colors.get(sel_glass_label, (255, 0, 255))
                cv2.rectangle(img_np, (x1, y1), (x2, y2), color, base_thickness)
                cv2.putText(img_np, sel_glass_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, color, base_thickness)

            for hand_box in hand_boxes:
                if hand_box is not None:
                    x1, y1, x2, y2 = map(int, hand_box)
                    color = colors["Hand"]
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, base_thickness)
                    cv2.putText(img_np, "Hand", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, color, base_thickness)

            for box_name in ["Box_IN", "Box_OUT", "Box_NG"]:
                box = self.last_boxes[box_name]
                if box is not None:
                    x1, y1, x2, y2 = map(int, box)
                    color = colors[box_name]
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), color, max(1, base_thickness // 2))

                    if box_name == "Box_OUT":
                        text_x = x1 - int(55 * scale_x)
                        text_y = y1 + int(10 * scale_y)
                    else:
                        text_x = x2 + int(4 * scale_x)
                        text_y = y1 + int(10 * scale_y)

                    cv2.putText(img_np, box_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, base_font_scale * 1.1, color, max(1, base_thickness // 2))

            self.draw_stats_overlay(img_np, scale_x, scale_y)
            return img_np, self.period

    def draw_stats_overlay(self, img: np.ndarray, scale_x: float = 1.0, scale_y: float = 1.0) -> None:
        """
        绘制统计叠加层
        参数:
          img: 原始帧
          scale_x/scale_y: 文本与线宽缩放因子
        """
        # 1. 设置叠加层起始位置 (左侧中间，稍微靠上)
        h, w = img.shape[:2]
        start_x, start_y = int(15 * scale_x), h // 2 - int(80 * scale_y)

        # 2. 绘制统计文本
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.44 * scale_x  # 调小为原 0.55 的 80%
        thickness = max(1, int(1 * scale_x))
        line_height = int(22 * scale_y)   # 紧凑间距
        
        curr_y = start_y
        
        # 定义显示模板
        display_data = [
            ("Pad Glass", (255, 255, 255)),
            (f"  OK: {self.cumulative_stats['Pad_Glass']['OK']}", (0, 255, 0)),
            (f"  NG: {self.cumulative_stats['Pad_Glass']['NG']}", (0, 0, 255)),
            (f"  LAST: {self.cumulative_stats['Pad_Glass']['LAST']:.2f}s" if isinstance(self.cumulative_stats['Pad_Glass']['LAST'], (int, float)) else "  LAST: -", (255, 255, 0)),
            ("", None),
            ("Phone Glass", (255, 255, 255)),
            (f"  OK: {self.cumulative_stats['Phone_Glass']['OK']}", (0, 255, 0)),
            (f"  NG: {self.cumulative_stats['Phone_Glass']['NG']}", (0, 0, 255)),
            (f"  LAST: {self.cumulative_stats['Phone_Glass']['LAST']:.2f}s" if isinstance(self.cumulative_stats['Phone_Glass']['LAST'], (int, float)) else "  LAST: -", (255, 255, 0)),
        ]

        for text, text_color in display_data:
            if text:
                # 仅保留主体文字，移除阴影
                cv2.putText(img, text, (start_x, curr_y), font, font_scale, text_color, thickness)
            curr_y += line_height

    def start_cycle(self) -> None:
        """
        周期开始处理：重置状态并记录起始帧与时间
        """
        self.in_cycle = True
        self.period = False
        self.has_left_start_box = False  # 重置离开标记
        self.current_glass_type = None   # 重置当前玻璃类型
        self.glass_type_votes = Counter() # 初始化类别投票
        self.cycle_start_frame = self.frame_count  # 记录起始帧
        self.cycle_start = time.time()
        self.orientation_timestamps = []
        self.current_orientation = None
        self.cycle_id = str(uuid.uuid4())  # 生成唯一ID
        self.glass_count += 1
        self.cycle_done = False

    def end_cycle(self) -> dict[str, float | int | str | None]:
        """
        周期结束处理
        返回:
          周期统计字典，包括时间戳/总时长/姿态时长/结果 等
        """
        self.in_cycle = False
        self.period = True
        self.last_boxes["Pad_Glass"] = None
        self.last_boxes["Phone_Glass"] = None
        self.cycle_done = True

        # 根据周期内的投票结果确定最终类别
        if self.glass_type_votes:
            self.current_glass_type = self.glass_type_votes.most_common(1)[0][0]

        # 更新累计统计数据
        if self.current_glass_type in self.cumulative_stats:
            res = self.current_box if self.current_box in ["OK", "NG"] else "OK"
            self.cumulative_stats[self.current_glass_type][res] += 1

        # 计算总时长
        # total_duration = time.time() - self.cycle_start
        total_frames = self.frame_count - self.cycle_start_frame
        total_duration = total_frames / self.fps  # 转换为秒数
        # 构建统计数据
        stats = {
            "id": self.cycle_id,
            "glass_count": self.glass_count,
            "time_stamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "horizontal": 0.0,
            "vertical": 0.0,
            "total_time": total_duration,
            "glass_result": self.current_box,
        }

        # 计算方向持续时间
        for i in range(1, len(self.orientation_timestamps)):
            prev_orient, prev_frame = self.orientation_timestamps[i - 1]
            curr_orient, curr_frame = self.orientation_timestamps[i]
            duration = (curr_frame - prev_frame) / self.fps
            stats[prev_orient] += duration

        self.history.append(stats)
        if self.current_glass_type in self.cumulative_stats:
            self.cumulative_stats[self.current_glass_type]["LAST"] = total_duration
        return stats

    def get_cycle_stats(self) -> dict[str, float | int | str | None] | None:
        """
        取出一条历史周期统计（先进先出）
        返回:
          周期统计或 None
        """
        return self.history.popleft() if self.history else None

    def process_orientation(self, glass_box: list) -> None:
        """
        记录当前帧玻璃姿态与帧编号
        参数:
          glass_box: 玻璃边界框（仅用于姿态推断）
        """
        new_orientation = self.detect_orientation(glass_box, self.box_ratio)
        self.orientation_timestamps.append((new_orientation, self.frame_count))
        # print("now orientation:", new_orientation)
        # if new_orientation != self.current_orientation:
        #     if self.current_orientation is not None:
        #         self.orientation_timestamps.append(
        #             (self.current_orientation, self.frame_count)
        #         )
        #     self.current_orientation = new_orientation

    # def get_cycle_stats(self):
    #     return self.history[-1] if self.history else None


def process_video_stream(video_path: str, output_path: str, model_path: str, yaml_path: str, static_boxes_path: str = "sg_boxes_config.json") -> None:
    """
    视频流处理入口
    参数:
      video_path: 视频路径或流地址
      output_path: 输出路径
      model_path: 模型权重路径
      static_boxes_path: 静态框配置路径
    流程:
      - 初始化检测器与写出器
      - 持续读取帧并进行业务处理与展示
      - 周期结束时输出统计
    """
    detectors = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 移除强制分辨率，使用原始视频分辨率以提高清晰度
    out_width, out_height = width, height
    
    # 设置输出目录
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    # 构建输出文件名
    video_name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(video_name)[0]
    output_path = os.path.join(os.path.dirname(output_path), f"{name_without_ext}_result.mp4")
    
    # 初始化 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    dataset_names = load_dataset_names(yaml_path)
    tracker = ObjectTracker(detectors, dataset_names, static_boxes_path=static_boxes_path, fps=fps)  # 修改构造函数

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, period_flag = tracker.process_frame(frame)
        
        # 写入帧
        out.write(annotated_frame)
        
        cv2.namedWindow("Video Stream", cv2.WINDOW_NORMAL)
        cv2.imshow("Video Stream", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if period_flag:
            while tracker.history:
                result = tracker.get_cycle_stats()
                print(
                    f"""
                ID: {result['id']}
                时间戳: {result['time_stamp']}
                检测玻璃数量: {result['glass_count']}
                检测结果：{result['glass_result']}
                总时长: {result['total_time']:.2f}s
                横向时长: {result['horizontal']:.2f}s
                纵向时长: {result['vertical']:.2f}s
                """
                )

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to: {output_path}")


if __name__ == "__main__":

    process_video_stream(
        video_path="videos/raw/phone-glass-BA远程13_L3-3F-318#照检查台_20260103075300_20260103103000.avi",
        output_path="videos/results/",
        model_path="D:/3_CODES/_YOLO/YOLO/runs/train/mimicry_yolov8n_20260121_bc12_epc300/weights/best.pt",
        yaml_path="config/data_config/mimicry_20260121_data.yaml",
        static_boxes_path="config/boxes_config/phone_boxes_config.json"
    )
