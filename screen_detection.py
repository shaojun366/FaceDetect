"""Screen detection, optional held-key mouse following, and latency logging."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from detectors import add_detector_arguments, create_detector, experiment_metadata
from mouse_control import KEYS, WindowsInput, movement_delta, prepare_dpi, select_target


WINDOW_NAME = "FaceDetect screen experiment"
FIELDS = ("frame", "started_ns", "capture_ms", "detect_ms", "decision_ms", "send_ms",
          "pipeline_ms", "input_latency_ms", "heads", "target_x", "target_y", "dx", "dy", "active")


def geometry(monitor):
    keys = ("left", "top", "width", "height")
    return {key: int(getattr(monitor, key) if hasattr(monitor, key) else monitor[key])
            for key in keys}


class ScreenSource:
    def __init__(self, monitor=1, region=None):
        self.monitor, self.region, self.capture = monitor, region, None

    def __enter__(self):
        from mss import MSS
        from mss.exception import ScreenShotError
        self.capture_error = ScreenShotError
        try:
            self.capture = MSS()
        except ScreenShotError as exc:
            raise RuntimeError(f"屏幕采集初始化失败，请检查显示器和系统屏幕录制权限：{exc}") from exc
        try:
            if not 1 <= self.monitor < len(self.capture.monitors):
                raise ValueError(f"显示器编号不存在：{self.monitor}，请使用 --list-monitors")
            monitor = geometry(self.capture.monitors[self.monitor])
            self.bounds = monitor.copy()
            if self.region:
                x, y, width, height = self.region
                if (min(x, y) < 0 or min(width, height) <= 0 or
                        x + width > monitor["width"] or y + height > monitor["height"]):
                    raise ValueError("--region 必须为所选显示器内部的相对区域：LEFT TOP WIDTH HEIGHT")
                self.bounds = {"left": monitor["left"] + x, "top": monitor["top"] + y,
                               "width": width, "height": height}
            self.origin = (monitor["left"] + monitor["width"] / 2 - self.bounds["left"],
                           monitor["top"] + monitor["height"] / 2 - self.bounds["top"])
            return self
        except BaseException:
            self.capture.close()
            raise

    def read(self):
        try:
            pixels = np.asarray(self.capture.grab(self.bounds))
        except self.capture_error as exc:
            raise RuntimeError(f"无法读取屏幕画面：{exc}") from exc
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)

    def __exit__(self, *_):
        self.capture.close()


def list_monitors():
    from mss import MSS
    with MSS() as source:
        return [{"index": index, **geometry(monitor)}
                for index, monitor in enumerate(source.monitors) if index > 0]


def exclude_preview_from_capture():
    if sys.platform == "win32":
        import ctypes
        api = ctypes.WinDLL("user32", use_last_error=True)
        api.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        api.FindWindowW.restype = ctypes.c_void_p
        api.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        window = api.FindWindowW(None, WINDOW_NAME)
        if not window or not api.SetWindowDisplayAffinity(window, 0x00000011):
            raise RuntimeError("无法从截图中排除预览窗口，请关闭 --preview 再运行")


def validate_options(args):
    values = (args.fps, args.radius, args.smoothing, args.deadzone, args.max_frame_age)
    if not all(np.isfinite(values)):
        raise ValueError("实验参数必须为有限数值")
    if (args.fps <= 0 or args.radius <= 0 or args.smoothing < 1 or args.deadzone < 0 or
            args.max_step < 1 or args.max_frame_age <= 0 or args.frames < 0):
        raise ValueError("帧率、范围、步长、帧龄必须为正数，平滑系数至少为 1，死区和帧数不能为负数")
    if args.aim and not args.window_title.strip():
        raise ValueError("启用鼠标跟随需要 --window-title 指定目标前台窗口的标题片段")
    if args.aim and sys.platform != "win32":
        raise ValueError("鼠标跟随目前仅支持 Windows；其他平台可运行检测与离线评测")


def process_frame(frame, detector, origin, args, controls, captured_window, started):
    detect_started = time.perf_counter()
    boxes = detector.detect(frame)
    detect_finished = time.perf_counter()
    target = select_target(boxes, origin, args.radius)
    dx, dy = movement_delta(target, origin, args.smoothing, args.max_step, args.deadzone)
    active = False
    if (args.aim and controls and not controls.pressed(0x77) and
            (time.perf_counter() - started) * 1000 <= args.max_frame_age):
        active = controls.allowed(args.window_title, KEYS[args.aim_key], captured_window)
    if not active:
        dx, dy = 0, 0
    decision_finished = time.perf_counter()
    if dx or dy:
        controls.move(dx, dy)
    finished = time.perf_counter()
    metrics = {"detect_ms": (detect_finished - detect_started) * 1000,
               "decision_ms": (decision_finished - detect_finished) * 1000,
               "send_ms": (finished - decision_finished) * 1000,
               "pipeline_ms": (finished - started) * 1000,
               "input_latency_ms": (finished - started) * 1000 if dx or dy else "",
               "heads": len(boxes), "target_x": target.center[0] if target else "",
               "target_y": target.center[1] if target else "", "dx": dx, "dy": dy,
               "active": int(active)}
    return boxes, target, metrics


def show_preview(frame, boxes, target, origin, args, metrics):
    for box in boxes:
        color = (0, 255, 255) if box is target else (255, 0, 0)
        cv2.rectangle(frame, (round(box.x), round(box.y)),
                      (round(box.x + box.width), round(box.y + box.height)), color, 2)
    center = tuple(round(value) for value in origin)
    cv2.circle(frame, center, round(args.radius), (0, 255, 0), 1)
    cv2.drawMarker(frame, center, (0, 255, 0), cv2.MARKER_CROSS, 20, 1)
    label = f"{args.detector} | {metrics['pipeline_ms']:.1f} ms | Heads: {len(boxes)}"
    cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if frame.shape[1] > 960:
        frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
    cv2.imshow(WINDOW_NAME, frame)


def run(args, detector):
    prepare_dpi()
    controls = WindowsInput() if sys.platform == "win32" else None
    log_path = Path(args.log or datetime.now().strftime("results/live-%Y%m%d-%H%M%S-%f.csv"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = experiment_metadata(args)
    metadata["started_utc"] = datetime.now(timezone.utc).isoformat()
    preview_created = False
    frames = 0
    try:
        with ScreenSource(args.monitor, args.region) as source, log_path.open(
                "x", encoding="utf-8", newline="") as log:
            metadata.update({"capture_bounds": source.bounds, "reticle_origin": source.origin})
            with log_path.with_suffix(".json").open("x", encoding="utf-8") as sidecar:
                json.dump(metadata, sidecar, ensure_ascii=False, indent=2)
            writer = csv.DictWriter(log, fieldnames=FIELDS)
            writer.writeheader()
            if args.preview:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
                preview_created = True
                exclude_preview_from_capture()
            print(f"屏幕实验已启动，日志：{log_path}。Ctrl+C 退出；Windows 可按 F8 全局退出。")
            while args.frames == 0 or frames < args.frames:
                if controls and controls.pressed(0x77):
                    break
                started = time.perf_counter()
                started_ns = time.perf_counter_ns()
                captured_window = controls.foreground()[0] if controls else None
                frame = source.read()
                captured = time.perf_counter()
                boxes, target, metrics = process_frame(
                    frame, detector, source.origin, args, controls, captured_window, started
                )
                frames += 1
                writer.writerow({"frame": frames, "started_ns": started_ns,
                                 "capture_ms": (captured - started) * 1000, **metrics})
                if frames % 60 == 0:
                    log.flush()
                if args.preview:
                    show_preview(frame, boxes, target, source.origin, args, metrics)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q"), ord("Q")):
                        break
                    if getattr(cv2, "currentUIFramework", lambda: "")() != "WAYLAND":
                        try:
                            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                                break
                        except cv2.error:
                            break
                time.sleep(max(0, 1 / args.fps - (time.perf_counter() - started)))
    finally:
        if preview_created:
            try:
                cv2.destroyWindow(WINDOW_NAME)
            except cv2.error:
                pass
    return frames


def build_parser():
    parser = argparse.ArgumentParser(description="屏幕人头检测、鼠标跟随与延迟实验")
    add_detector_arguments(parser)
    parser.add_argument("--list-monitors", action="store_true")
    parser.add_argument("--monitor", type=int, default=1)
    parser.add_argument("--region", nargs=4, type=int, metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"))
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--frames", type=int, default=0, help="采样帧数，0 表示持续运行")
    parser.add_argument("--preview", action="store_true", help="显示检测窗口；默认关闭")
    parser.add_argument("--log", help="CSV 路径，已有文件不会被覆盖")
    parser.add_argument("--aim", action="store_true", help="启用按键按住时的鼠标跟随")
    parser.add_argument("--window-title", default="", help="允许鼠标跟随的前台窗口标题片段")
    parser.add_argument("--aim-key", choices=tuple(KEYS), default="right")
    parser.add_argument("--radius", type=float, default=180, help="准星附近的目标选择半径，像素")
    parser.add_argument("--smoothing", type=float, default=5)
    parser.add_argument("--max-step", type=int, default=25, help="每帧每轴最大相对鼠标位移")
    parser.add_argument("--deadzone", type=float, default=2)
    parser.add_argument("--max-frame-age", type=float, default=100, help="允许发出鼠标指令的最大帧龄，毫秒")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.list_monitors:
            prepare_dpi()
            print(json.dumps(list_monitors(), ensure_ascii=False, indent=2))
            return 0
        validate_options(args)
        run(args, create_detector(args))
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, OSError, ImportError, cv2.error) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
