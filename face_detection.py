"""Real-time Haar cascade face detection. Run with --help for options."""

import argparse
import math
from pathlib import Path
import sys
import time

import cv2


WINDOW_NAME = "FaceDetect"


def load_cascade(path=None):
    """Load the bundled frontal-face classifier or a custom cascade."""
    cascade_path = Path(path) if path else (
        Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    )
    if not cascade_path.is_file():
        raise RuntimeError(f"找不到人脸分类器：{cascade_path}")
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"无法加载人脸分类器：{cascade_path}")
    return cascade


def detect_faces(frame, cascade, process_width=320, scale_factor=1.2,
                 min_neighbors=3, min_size=20):
    """Detect on a small image, returning boxes in the original frame's pixels."""
    height, width = frame.shape[:2]
    target_width = min(process_width, width)
    target_height = max(1, round(height * target_width / width))
    small = cv2.resize(frame, (target_width, target_height),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
        minSize=(min_size, min_size), flags=cv2.CASCADE_SCALE_IMAGE,
    )
    scale_x, scale_y = width / target_width, height / target_height
    return [
        (round(x * scale_x), round(y * scale_y),
         round(w * scale_x), round(h * scale_y))
        for x, y, w, h in faces
    ]


def video_demo(camera=0, width=640, height=480, process_width=320,
               scale_factor=1.2, min_neighbors=3, min_size=20,
               mirror=True, cascade_path=None):
    """Display camera frames; always release the camera and preview window."""
    if camera < 0 or min_neighbors < 0:
        raise ValueError("摄像头编号和 min-neighbors 不能为负数")
    if min(width, height, process_width, min_size) <= 0:
        raise ValueError("分辨率、处理宽度和最小人脸尺寸必须大于 0")
    if not math.isfinite(scale_factor) or scale_factor <= 1:
        raise ValueError("scale-factor 必须为大于 1 的有限数值")

    cascade = load_cascade(cascade_path)
    capture = cv2.VideoCapture(camera)
    window_created = False
    try:
        if not capture.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera}，请检查设备、权限或 --camera 参数")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        window_created = True
        # Wayland does not implement getWindowProperty; keyboard exit still works.
        check_visibility = getattr(cv2, "currentUIFramework", lambda: "")() != "WAYLAND"
        previous_time = time.perf_counter()
        frame_count, fps = 0, 0.0

        while True:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError("无法读取摄像头画面，请检查设备连接或是否被其他程序占用")
            faces = detect_faces(frame, cascade, process_width, scale_factor,
                                 min_neighbors, min_size)
            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            if mirror:
                frame = cv2.flip(frame, 1)

            frame_count += 1
            if frame_count >= 30:
                current_time = time.perf_counter()
                elapsed = current_time - previous_time
                if elapsed > 0:
                    fps = frame_count / elapsed
                previous_time, frame_count = current_time, 0
            # Draw the last measured FPS on every frame, including warm-up frames.
            cv2.putText(frame, f"FPS: {fps:.2f}  Faces: {len(faces)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                break
            if check_visibility:
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    # Some desktop backends raise after the user closes the window.
                    break
    finally:
        capture.release()
        if window_created:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass  # A closed window must not hide the original processing error.


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为大于 0 的整数")
    return number


def _nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("必须为非负整数")
    return number


def _scale_factor(value):
    number = float(value)
    if not math.isfinite(number) or number <= 1:
        raise argparse.ArgumentTypeError("必须为大于 1 的有限数值")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description="使用 OpenCV 实时检测摄像头中的人脸")
    parser.add_argument("--camera", type=_nonnegative_int, default=0, help="摄像头编号（默认 0）")
    parser.add_argument("--width", type=_positive_int, default=640, help="请求的采集宽度")
    parser.add_argument("--height", type=_positive_int, default=480, help="请求的采集高度")
    parser.add_argument("--process-width", type=_positive_int, default=320,
                        help="检测图像宽度，保持比例且不放大原图（默认 320）")
    parser.add_argument("--scale-factor", type=_scale_factor, default=1.2,
                        help="级联检测的尺度步长，必须大于 1（默认 1.2）")
    parser.add_argument("--min-neighbors", type=_nonnegative_int, default=3,
                        help="候选框的最少邻居数（默认 3）")
    parser.add_argument("--min-size", type=_positive_int, default=20,
                        help="检测图像中最小人脸的边长，单位像素（默认 20）")
    parser.add_argument("--no-mirror", action="store_false", dest="mirror",
                        help="关闭镜像显示")
    parser.add_argument("--cascade", dest="cascade_path", help="自定义 Haar 分类器 XML 路径")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        video_demo(**vars(args))
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError, cv2.error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
