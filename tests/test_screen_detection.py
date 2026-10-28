"""No screen capture or physical mouse input occurs in these tests."""

from contextlib import redirect_stderr, redirect_stdout
import ctypes
import hashlib
from importlib.util import find_spec
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import benchmark_detection as benchmark
import detectors
import download_model
from detectors import Detection, YoloDetector
import mouse_control as mouse
import screen_detection as screen


FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_yolov8.onnx"


class YoloTests(unittest.TestCase):
    def setUp(self):
        with patch.object(cv2.dnn, "readNetFromONNX"):
            self.detector = YoloDetector(FIXTURE)

    def test_letterbox_keeps_geometry_and_padding(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        image, transform = detectors.letterbox(frame, 640)
        self.assertEqual(image.shape, (640, 640, 3))
        self.assertEqual(transform, (0.5, 0.5, 0, 140))
        self.assertTrue(np.all(image[0] == 114))
        self.assertTrue(np.all(image[140:500] == 0))

    def test_v8_coordinates_remove_letterbox_padding(self):
        output = np.array([[[320], [320], [40], [40], [0.9]]], np.float32)
        boxes = self.detector.decode(output, (720, 1280, 3), (0.5, 0.5, 0, 140))
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].center, (640, 360))
        self.assertEqual((boxes[0].width, boxes[0].height), (80, 80))

    def test_v5_uses_objectness_times_class_score(self):
        self.detector.output_format = "v5"
        output = np.array([[[20, 20, 10, 10, 0.8, 0.9], [50, 50, 10, 10, 0.5, 0.9]]])
        boxes = self.detector.decode(output, (100, 100), (1, 1, 0, 0))
        self.assertEqual(len(boxes), 1)
        self.assertAlmostEqual(boxes[0].confidence, 0.72)

    def test_end2end_format(self):
        self.detector.output_format = "end2end"
        boxes = self.detector.decode(np.array([[[10, 20, 30, 40, 0.9, 0]]]),
                                     (100, 100), (1, 1, 0, 0))
        self.assertEqual(boxes[0].center, (20, 30))

    def test_nms_removes_duplicate_boxes(self):
        output = np.array([[[50, 51], [50, 50], [20, 20], [20, 20], [0.9, 0.8]]])
        boxes = self.detector.decode(output, (100, 100), (1, 1, 0, 0))
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].confidence, 0.9)

    def test_nms_preserves_different_classes(self):
        self.detector.class_ids = {0, 1}
        output = np.array([[[50, 50], [50, 50], [20, 20], [20, 20], [0.9, 0.1], [0.1, 0.9]]])
        boxes = self.detector.decode(output, (100, 100), (1, 1, 0, 0))
        self.assertEqual({box.class_id for box in boxes}, {0, 1})

    def test_invalid_and_unselected_detections_are_dropped(self):
        self.detector.output_format = "end2end"
        output = np.array([[[1, 2, 3, 4, 0.9, 1], [1, 2, 3, 4, 0.2, 0],
                            [1, 2, 3, 4, float("nan"), 0], [4, 2, 1, 4, 0.9, 0]]])
        self.assertEqual(self.detector.decode(output, (100, 100), (1, 1, 0, 0)), [])

    def test_invalid_output_shape_and_class_are_reported(self):
        for output in [np.zeros((5, 8400)), np.zeros((2, 5, 1)), np.zeros((1, 4, 1))]:
            with self.subTest(shape=output.shape), self.assertRaises(ValueError):
                self.detector.decode(output, (100, 100), (1, 1, 0, 0))

    def test_actual_opencv_onnx_inference(self):
        detector = YoloDetector(FIXTURE)
        boxes = detector.detect(np.zeros((720, 1280, 3), np.uint8))
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].center, (640, 360))

    @unittest.skipUnless(find_spec("onnxruntime"), "optional ONNX Runtime is not installed")
    def test_actual_onnxruntime_matches_opencv(self):
        frame = np.zeros((720, 1280, 3), np.uint8)
        opencv_boxes = YoloDetector(FIXTURE).detect(frame)
        runtime_boxes = YoloDetector(FIXTURE, backend="onnxruntime").detect(frame)
        self.assertEqual(opencv_boxes, runtime_boxes)


class MouseTests(unittest.TestCase):
    def test_nearest_head_inside_radius_is_selected(self):
        near, far = Detection(110, 90, 20, 20), Detection(200, 90, 20, 20)
        self.assertIs(mouse.select_target([far, near], (100, 100), 50), near)
        self.assertIsNone(mouse.select_target([far], (100, 100), 50))

    def test_movement_has_deadzone_smoothing_and_per_axis_limit(self):
        self.assertEqual(mouse.movement_delta(None, (100, 100)), (0, 0))
        self.assertEqual(mouse.movement_delta(Detection(99, 99, 2, 2), (100, 100)), (0, 0))
        self.assertEqual(mouse.movement_delta(Detection(200, 0, 20, 20), (100, 100), 2, 10), (10, -10))
        self.assertEqual(mouse.movement_delta(Detection(102, 99, 2, 2), (100, 100), 20), (1, 0))

    def test_windows_input_structure_and_move_only_flags(self):
        api = MagicMock()
        api.SendInput.return_value = 1
        controls = mouse.WindowsInput(api)
        controls.move(12, -7)
        count, pointer, size = api.SendInput.call_args.args
        event = ctypes.cast(pointer, ctypes.POINTER(mouse._Input)).contents
        self.assertEqual(count, 1)
        self.assertEqual(size, 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28)
        self.assertEqual((event.mi.dx, event.mi.dy, event.mi.dwFlags), (12, -7, 1))
        self.assertEqual(event.type, 0)

    def test_failed_input_is_reported(self):
        api = MagicMock()
        api.SendInput.return_value = 0
        with self.assertRaisesRegex(RuntimeError, "鼠标输入发送失败"):
            mouse.WindowsInput(api).move(1, 1)

    def test_key_state_uses_current_high_bit(self):
        api = MagicMock()
        controls = mouse.WindowsInput(api)
        api.GetAsyncKeyState.return_value = 1
        self.assertFalse(controls.pressed(2))
        api.GetAsyncKeyState.return_value = -32768
        self.assertTrue(controls.pressed(2))

    def test_foreground_and_key_are_both_required(self):
        controls = mouse.WindowsInput(MagicMock())
        with patch.object(controls, "foreground", return_value=(123, "Research Demo")), \
                patch.object(controls, "pressed", return_value=True):
            self.assertTrue(controls.allowed("demo", 2, 123))
            self.assertFalse(controls.allowed("demo", 2, 456))
            self.assertFalse(controls.allowed("other", 2, 123))
            self.assertFalse(controls.allowed("", 2, 123))


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.args = screen.build_parser().parse_args(["--aim", "--window-title", "Demo"])
        self.detector = MagicMock()
        self.detector.detect.return_value = [Detection(110, 90, 20, 20)]
        self.controls = MagicMock()
        self.controls.pressed.return_value = False
        self.controls.allowed.return_value = True

    def process(self, started=99.99):
        with patch.object(screen.time, "perf_counter", return_value=100.0):
            return screen.process_frame(np.zeros((200, 200, 3), np.uint8), self.detector,
                                        (100, 100), self.args, self.controls, 123, started)

    def test_enabled_and_held_key_sends_relative_move(self):
        _, _, metrics = self.process()
        self.controls.move.assert_called_once_with(4, 0)
        self.assertAlmostEqual(metrics["input_latency_ms"], 10)

    def test_observation_mode_never_moves(self):
        self.args.aim = False
        _, _, metrics = self.process()
        self.controls.move.assert_not_called()
        self.assertEqual(metrics["input_latency_ms"], "")

    def test_released_key_or_changed_window_never_moves(self):
        self.controls.allowed.return_value = False
        self.process()
        self.controls.move.assert_not_called()

    def test_stale_frame_and_emergency_key_never_move(self):
        self.process(started=90)
        self.controls.move.assert_not_called()
        self.controls.pressed.return_value = True
        self.process()
        self.controls.move.assert_not_called()

    def test_missing_target_never_moves(self):
        self.detector.detect.return_value = []
        self.process()
        self.controls.move.assert_not_called()

    def test_screen_region_offsets_and_bgra_conversion(self):
        capture = MagicMock()
        capture.monitors = [{}, {"left": -1920, "top": 0, "width": 1920, "height": 1080}]
        capture.grab.return_value = np.full((200, 300, 4), [1, 2, 3, 255], np.uint8)
        with patch("mss.MSS", return_value=capture):
            with screen.ScreenSource(1, (100, 50, 300, 200)) as source:
                self.assertEqual(source.bounds["left"], -1820)
                self.assertEqual(source.origin, (860, 490))
                self.assertEqual(source.read()[0, 0].tolist(), [1, 2, 3])
        capture.close.assert_called_once_with()

    def test_invalid_region_closes_capture(self):
        capture = MagicMock()
        capture.monitors = [{}, {"left": 0, "top": 0, "width": 100, "height": 100}]
        with patch("mss.MSS", return_value=capture), self.assertRaises(ValueError):
            with screen.ScreenSource(1, (90, 0, 20, 20)):
                pass
        capture.close.assert_called_once_with()

    def test_capture_failure_is_reported_and_resources_are_closed(self):
        from mss.exception import ScreenShotError
        capture = MagicMock()
        capture.monitors = [{}, {"left": 0, "top": 0, "width": 100, "height": 100}]
        capture.grab.side_effect = ScreenShotError("capture failed")
        with patch("mss.MSS", return_value=capture), self.assertRaisesRegex(RuntimeError, "无法读取屏幕"):
            with screen.ScreenSource(1) as source:
                source.read()
        capture.close.assert_called_once_with()

    def test_invalid_parameters_are_rejected(self):
        for options in [["--fps", "nan"], ["--smoothing", "0"], ["--frames", "-1"],
                        ["--aim"], ["--deadzone", "-1"]]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                screen.validate_options(screen.build_parser().parse_args(options))

    def test_live_loop_writes_csv_and_metadata_without_real_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "live.csv"
            args = screen.build_parser().parse_args(["--frames", "2", "--log", str(path)])
            source = MagicMock()
            source.bounds = {"left": 0, "top": 0, "width": 200, "height": 200}
            source.origin = (100, 100)
            source.read.return_value = np.zeros((200, 200, 3), np.uint8)
            with patch.object(screen.sys, "platform", "linux"), \
                    patch.object(screen, "prepare_dpi"), \
                    patch.object(screen, "ScreenSource") as factory, \
                    patch.object(screen, "experiment_metadata", return_value={}), \
                    patch.object(screen.time, "sleep"), redirect_stdout(io.StringIO()):
                factory.return_value.__enter__.return_value = source
                self.assertEqual(screen.run(args, self.detector), 2)
            import csv
            with path.open(encoding="utf-8", newline="") as output:
                rows = list(csv.DictReader(output))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["dx"], "0")
            self.assertEqual(rows[0]["input_latency_ms"], "")
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["reticle_origin"], [100, 100])
            factory.return_value.__exit__.assert_called_once()


class BenchmarkTests(unittest.TestCase):
    def test_matching_is_one_to_one_and_counts_false_positives(self):
        box = Detection(10, 10, 20, 20)
        self.assertEqual(benchmark.match_predictions([box, box], [box]), (1, 1, 0))
        self.assertEqual(benchmark.match_predictions([], [box]), (0, 0, 1))
        self.assertEqual(benchmark.match_predictions([box], []), (0, 1, 0))
        self.assertEqual(benchmark.intersection_over_union(box, box), 1)

    def test_summary_has_percentiles_and_quality(self):
        summary = benchmark.summarize([1, 2, 3, 4, 5], (2, 1, 2))
        self.assertEqual(summary["p50_ms"], 3)
        self.assertAlmostEqual(summary["p95_ms"], 4.8)
        self.assertAlmostEqual(summary["precision"], 2 / 3)
        self.assertEqual(summary["recall"], 0.5)

    def test_manifest_rejects_invalid_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.json"
            for manifest in [[], {}, {"schema_version": 1, "images": "invalid"}]:
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                    benchmark.load_manifest(path)

    def test_actual_manifest_to_report_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            ok, data = cv2.imencode(".png", np.zeros((640, 640, 3), np.uint8))
            self.assertTrue(ok)
            (folder / "frame.png").write_bytes(data.tobytes())
            manifest = {"schema_version": 1, "images": [
                {"file": "frame.png", "game": "synthetic", "style": "fixture",
                 "boxes": [[300, 300, 40, 40]]}]}
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            arguments = ["--manifest", str(folder / "manifest.json"), "--detector", "yolo",
                         "--model", str(FIXTURE), "--repeat", "2", "--warmup", "1",
                         "--output", str(folder / "report")]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark.main(arguments), 0)
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["overall"]["tp"], 1)
            self.assertEqual(report["overall"]["samples"], 2)
            self.assertEqual(report["groups"][0]["style"], "fixture")
            self.assertIn("model_sha256", report)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(benchmark.main(arguments), 1)

    def test_quality_counts_once_per_image_and_groups_by_style(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "blank.png"
            _, data = cv2.imencode(".png", np.zeros((20, 20, 3), np.uint8))
            path.write_bytes(data.tobytes())
            records = [{"path": path, "file": "blank.png", "game": "a", "style": style,
                        "truth": [Detection(1, 1, 10, 10)]} for style in ["anime", "realistic"]]
            detector = MagicMock()
            detector.detect.return_value = [Detection(1, 1, 10, 10)]
            _, summary = benchmark.evaluate(detector, records, repeat=3, warmup=2)
            self.assertEqual(summary["overall"]["tp"], 2)
            self.assertEqual(summary["overall"]["samples"], 6)
            self.assertEqual(len(summary["groups"]), 2)
            self.assertEqual(detector.detect.call_count, 8)


class DownloadTests(unittest.TestCase):
    def test_hash_verified_download_and_existing_file_protection(self):
        data = b"synthetic model bytes"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "head.onnx"
            with patch.object(download_model, "SIZE", len(data)), \
                    patch.object(download_model, "SHA256", hashlib.sha256(data).hexdigest()), \
                    patch.object(download_model, "urlopen") as request:
                request.return_value.__enter__.return_value.read.return_value = data
                self.assertEqual(download_model.download(path), path)
                self.assertEqual(path.read_bytes(), data)
                with self.assertRaises(ValueError):
                    download_model.download(path)
                self.assertEqual(request.call_count, 1)

    def test_invalid_download_is_not_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "head.onnx"
            with patch.object(download_model, "urlopen") as request:
                request.return_value.__enter__.return_value.read.return_value = b"invalid"
                with self.assertRaises(ValueError):
                    download_model.download(path)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
