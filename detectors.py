"""Shared image-coordinate detector interface for live and offline experiments."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import cv2
import numpy as np

from face_detection import detect_faces, load_cascade


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    width: float
    height: float
    confidence: float = 1.0
    class_id: int = 0

    @property
    def center(self):
        return self.x + self.width / 2, self.y + self.height / 2


class HaarDetector:
    """Frontal-face baseline; Haar scores are not calibrated probabilities."""

    def __init__(self, process_width=640):
        self.cascade = load_cascade()
        self.process_width = process_width

    def detect(self, frame):
        return [Detection(*box) for box in
                detect_faces(frame, self.cascade, process_width=self.process_width)]


def letterbox(frame, size):
    height, width = frame.shape[:2]
    if min(height, width) <= 0:
        raise ValueError("输入图像不能为空")
    gain = min(size / width, size / height)
    resized_width = max(1, round(width * gain))
    resized_height = max(1, round(height * gain))
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    canvas = np.full((size, size, 3), 114, np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    return canvas, (resized_width / width, resized_height / height, left, top)


class YoloDetector:
    """CPU OpenCV DNN for batch-one, decoded FP32 YOLO ONNX detection exports.

    v8: [1, 4+C, N] cx/cy/w/h + class scores; no graph NMS.
    v5: [1, N, 5+C] cx/cy/w/h + objectness + class scores; no graph NMS.
    end2end: [1, N, 6] x1/y1/x2/y2/score/class, in input-image pixels.
    """

    def __init__(self, model, size=640, confidence=0.5, nms=0.45,
                 class_ids=(0,), output_format="v8", backend="opencv", threads=1):
        if not Path(model).is_file():
            raise ValueError(f"找不到 ONNX 模型：{model}")
        if size <= 0 or not 0 < confidence <= 1 or not 0 < nms <= 1:
            raise ValueError("模型尺寸和置信度/NMS 阈值无效")
        if output_format not in ("v8", "v5", "end2end"):
            raise ValueError("不支持的 YOLO 输出格式")
        if not class_ids or any(index < 0 for index in class_ids):
            raise ValueError("必须指定非负的头部类别编号")
        self.backend = backend
        if backend == "opencv":
            self.net = cv2.dnn.readNetFromONNX(str(model))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        elif backend == "onnxruntime":
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ValueError("请先安装可选依赖：python -m pip install -r requirements-yolo.txt") from exc
            ort.disable_telemetry_events()
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            try:
                self.session = ort.InferenceSession(str(model), sess_options=options,
                                                    providers=["CPUExecutionProvider"])
            except Exception as exc:
                raise RuntimeError(f"ONNX Runtime 无法加载模型：{exc}") from exc
            inputs = self.session.get_inputs()
            if len(inputs) != 1 or inputs[0].type != "tensor(float)":
                raise ValueError("模型必须使用单个 FP32 图像输入")
            expected = (1, 3, size, size)
            if len(inputs[0].shape) != 4 or any(
                    isinstance(actual, int) and actual != wanted
                    for actual, wanted in zip(inputs[0].shape, expected)):
                raise ValueError("模型输入维度与 batch=1、RGB 和 --input-size 不匹配")
            self.input_name = inputs[0].name
        else:
            raise ValueError("不支持的推理后端")
        self.size, self.confidence, self.nms = size, confidence, nms
        self.class_ids, self.output_format = set(class_ids), output_format

    def decode(self, output, frame_shape, transform):
        output = np.asarray(output)
        if output.ndim != 3 or output.shape[0] != 1:
            raise ValueError(f"不支持的模型输出形状 {output.shape}；请使用 batch=1 的检测模型")
        rows = output[0].T if self.output_format == "v8" else output[0]
        base = 5 if self.output_format == "v5" else 4
        if self.output_format == "end2end":
            if rows.shape[1] != 6:
                raise ValueError("end2end 输出必须为 [1, N, 6]")
        elif rows.shape[1] <= base or max(self.class_ids) >= rows.shape[1] - base:
            raise ValueError("模型类别数或输出格式与 --class-id / --yolo-format 不匹配")
        # Filter the thousands of background anchors in NumPy before boxing/NMS.
        rows = rows[np.isfinite(rows).all(axis=1)]
        if self.output_format == "end2end":
            scores, labels = rows[:, 4], rows[:, 5]
        else:
            labels = np.argmax(rows[:, base:], axis=1)
            scores = rows[np.arange(len(rows)), base + labels]
            if self.output_format == "v5":
                scores = scores * rows[:, 4]
        keep = (scores >= self.confidence) & (scores <= 1) & np.isin(labels, list(self.class_ids))
        rows, scores, labels = rows[keep], scores[keep], labels[keep]
        scale_x, scale_y, left, top = transform
        height, width = frame_shape[:2]
        candidates = []
        for row, score, label in zip(rows, scores, labels):
            if self.output_format == "end2end":
                x1, y1, x2, y2 = map(float, row[:4])
                if label != int(label):
                    continue
            else:
                cx, cy, box_width, box_height = map(float, row[:4])
                x1, y1 = cx - box_width / 2, cy - box_height / 2
                x2, y2 = cx + box_width / 2, cy + box_height / 2
            score, label = float(score), int(label)
            x1, x2 = np.clip([(x1 - left) / scale_x, (x2 - left) / scale_x], 0, width)
            y1, y2 = np.clip([(y1 - top) / scale_y, (y2 - top) / scale_y], 0, height)
            if x2 > x1 and y2 > y1:
                candidates.append(Detection(float(x1), float(y1), float(x2 - x1),
                                            float(y2 - y1), score, label))
        kept = []
        for label in sorted(self.class_ids):
            group = [box for box in candidates if box.class_id == label]
            if not group:
                continue
            indices = cv2.dnn.NMSBoxes(
                [[box.x, box.y, box.width, box.height] for box in group],
                [box.confidence for box in group], self.confidence, self.nms,
            )
            kept.extend(group[int(index)] for index in np.asarray(indices).reshape(-1))
        return sorted(kept, key=lambda box: box.confidence, reverse=True)

    def detect(self, frame):
        image, transform = letterbox(frame, self.size)
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.size, self.size),
                                     swapRB=True, crop=False)
        if self.backend == "opencv":
            self.net.setInput(blob)
            output = self.net.forward()
        else:
            try:
                output = self.session.run(None, {self.input_name: blob})[0]
            except Exception as exc:
                raise RuntimeError(f"ONNX Runtime 推理失败：{exc}") from exc
        return self.decode(output, frame.shape, transform)


def add_detector_arguments(parser):
    parser.add_argument("--detector", choices=("haar", "yolo"), default="haar")
    parser.add_argument("--model", help="已训练的人头检测 ONNX 模型路径")
    parser.add_argument("--backend", choices=("opencv", "onnxruntime"), default="opencv")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--class-id", type=int, nargs="+", default=[0])
    parser.add_argument("--yolo-format", choices=("v8", "v5", "end2end"), default="v8")
    parser.add_argument("--threads", type=int, default=1, help="OpenCV CPU 线程数")


def create_detector(args):
    if args.threads <= 0 or args.input_size <= 0:
        raise ValueError("线程数和输入尺寸必须大于 0")
    cv2.setNumThreads(args.threads)
    if args.detector == "haar":
        return HaarDetector(args.input_size)
    if not args.model:
        raise ValueError("YOLO 模式需要 --model 指定已训练的人头检测模型")
    return YoloDetector(args.model, args.input_size, args.confidence, args.nms,
                        args.class_id, args.yolo_format, args.backend, args.threads)


def experiment_metadata(args):
    import platform
    import sys
    model = Path(args.model) if args.detector == "yolo" and args.model else (
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    )
    metadata = {"parameters": vars(args), "python": sys.version, "platform": platform.platform(),
                "opencv": cv2.__version__, "numpy": np.__version__,
                "backend": "OpenCV CPU", "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    if args.detector == "yolo" and args.backend == "onnxruntime":
        from importlib.metadata import version
        metadata.update(backend="ONNX Runtime CPU", onnxruntime=version("onnxruntime"))
    return metadata
