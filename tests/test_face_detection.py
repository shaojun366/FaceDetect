"""Regression tests without opening a physical camera or GUI window."""

from contextlib import ExitStack, redirect_stderr
import importlib
import io
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

import face_detection as fd


class DetectionTests(unittest.TestCase):
    def test_import_does_not_open_camera(self):
        with patch.object(cv2, "VideoCapture") as camera:
            importlib.reload(fd)
        camera.assert_not_called()

    def test_bundled_classifier_runs_on_real_opencv(self):
        cascade = fd.load_cascade()
        self.assertFalse(cascade.empty())
        self.assertEqual(fd.detect_faces(np.zeros((480, 640, 3), np.uint8), cascade), [])

    def test_missing_classifier_fails_before_opening_camera(self):
        with patch.object(fd.Path, "is_file", return_value=False), \
                patch.object(cv2, "VideoCapture") as camera:
            with self.assertRaisesRegex(RuntimeError, "找不到人脸分类器"):
                fd.video_demo(cascade_path="missing.xml")
        camera.assert_not_called()

    def test_empty_classifier_reports_error(self):
        with patch.object(fd.Path, "is_file", return_value=True), \
                patch.object(cv2, "CascadeClassifier") as classifier:
            classifier.return_value.empty.return_value = True
            with self.assertRaisesRegex(RuntimeError, "无法加载人脸分类器"):
                fd.load_cascade("empty.xml")

    def test_widescreen_aspect_ratio_and_boxes_are_preserved(self):
        cascade = MagicMock()
        cascade.detectMultiScale.return_value = np.array([[10, 20, 30, 40]], np.int32)
        boxes = fd.detect_faces(np.zeros((720, 1280, 3), np.uint8), cascade)
        gray = cascade.detectMultiScale.call_args.args[0]
        self.assertEqual(gray.shape, (180, 320))
        self.assertEqual(boxes, [(40, 80, 120, 160)])

    def test_small_images_are_not_upscaled(self):
        cascade = MagicMock()
        cascade.detectMultiScale.return_value = [(10, 20, 30, 40)]
        boxes = fd.detect_faces(np.zeros((90, 160, 3), np.uint8), cascade)
        self.assertEqual(cascade.detectMultiScale.call_args.args[0].shape, (90, 160))
        self.assertEqual(boxes, [(10, 20, 30, 40)])

    def test_rounding_uses_actual_resized_height(self):
        cascade = MagicMock()
        cascade.detectMultiScale.return_value = [(10, 20, 30, 40)]
        boxes = fd.detect_faces(np.zeros((481, 853, 3), np.uint8), cascade)
        gray = cascade.detectMultiScale.call_args.args[0]
        self.assertEqual(gray.shape, (180, 320))
        self.assertEqual(boxes, [(27, 53, 80, 107)])


class CameraLoopTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.capture = MagicMock()
        self.capture.isOpened.return_value = True
        self.frame = np.zeros((480, 640, 3), np.uint8)
        self.capture.read.return_value = (True, self.frame)
        self.camera = stack.enter_context(patch.object(cv2, "VideoCapture", return_value=self.capture))
        stack.enter_context(patch.object(fd, "load_cascade"))
        self.detect = stack.enter_context(patch.object(fd, "detect_faces", return_value=[]))
        self.window = stack.enter_context(patch.object(cv2, "namedWindow"))
        self.destroy = stack.enter_context(patch.object(cv2, "destroyAllWindows"))
        self.show = stack.enter_context(patch.object(cv2, "imshow"))
        self.text = stack.enter_context(patch.object(cv2, "putText"))
        self.key = stack.enter_context(patch.object(cv2, "waitKey", return_value=27))
        self.visible = stack.enter_context(patch.object(cv2, "getWindowProperty", return_value=1))
        self.backend = stack.enter_context(
            patch.object(cv2, "currentUIFramework", return_value="WIN32", create=True)
        )

    def assert_released(self, window=True):
        self.capture.release.assert_called_once_with()
        if window:
            self.destroy.assert_called_once_with()
        else:
            self.destroy.assert_not_called()

    def test_camera_open_failure_is_an_error_and_releases_capture(self):
        self.capture.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, "无法打开摄像头"):
            fd.video_demo(camera=2)
        self.camera.assert_called_once_with(2)
        self.assert_released(window=False)

    def test_frame_read_failure_releases_resources(self):
        for result in [(False, None), (True, None), (True, np.empty((0, 0, 3), np.uint8))]:
            with self.subTest(result=result[0]):
                self.capture.release.reset_mock()
                self.destroy.reset_mock()
                self.capture.read.return_value = result
                with self.assertRaisesRegex(RuntimeError, "无法读取摄像头画面"):
                    fd.video_demo()
                self.assert_released()

    def test_processing_exception_releases_resources(self):
        self.detect.side_effect = cv2.error("detector failed")
        with self.assertRaises(cv2.error):
            fd.video_demo()
        self.assert_released()

    def test_preview_creation_failure_releases_capture(self):
        self.window.side_effect = cv2.error("GUI unavailable")
        with self.assertRaises(cv2.error):
            fd.video_demo()
        self.assert_released(window=False)

    def test_ctrl_c_releases_resources(self):
        self.capture.read.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            fd.video_demo()
        self.assert_released()

    def test_escape_and_q_exit_cleanly(self):
        for key in [27, ord("q"), ord("Q")]:
            with self.subTest(key=key):
                self.capture.release.reset_mock()
                self.destroy.reset_mock()
                self.key.return_value = key
                fd.video_demo()
                self.assert_released()

    def test_closed_window_exits(self):
        self.key.return_value = -1
        self.visible.return_value = 0
        fd.video_demo()
        self.assert_released()

    def test_backend_error_after_window_close_exits(self):
        self.key.return_value = -1
        self.visible.side_effect = cv2.error("window was closed")
        fd.video_demo()
        self.assert_released()

    def test_wayland_does_not_query_unsupported_window_property(self):
        self.backend.return_value = "WAYLAND"
        self.key.side_effect = [-1, 27]
        fd.video_demo()
        self.visible.assert_not_called()
        self.assert_released()

    def test_fps_is_drawn_on_every_frame_and_kept_after_update(self):
        self.key.side_effect = [-1] * 30 + [27]
        with patch.object(fd.time, "perf_counter", side_effect=[10.0, 11.0]):
            fd.video_demo()
        labels = [call.args[1] for call in self.text.call_args_list]
        self.assertEqual(len(labels), 31)
        self.assertEqual(labels[0], "FPS: 0.00  Faces: 0")
        self.assertEqual(labels[29], "FPS: 30.00  Faces: 0")
        self.assertEqual(labels[30], labels[29])
        self.assert_released()

    def test_fps_handles_zero_elapsed_time(self):
        self.key.side_effect = [-1] * 29 + [27]
        with patch.object(fd.time, "perf_counter", return_value=10.0):
            fd.video_demo()
        self.assertEqual(self.text.call_args.args[1], "FPS: 0.00  Faces: 0")
        self.assert_released()

    def test_no_mirror_preserves_original_frame(self):
        with patch.object(cv2, "flip") as flip:
            fd.video_demo(mirror=False)
        flip.assert_not_called()
        self.assertIs(self.show.call_args.args[1], self.frame)

    def test_cleanup_error_does_not_hide_processing_error(self):
        self.detect.side_effect = RuntimeError("original error")
        self.destroy.side_effect = cv2.error("cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "original error"):
            fd.video_demo()
        self.assert_released()

    def test_direct_invalid_options_do_not_open_camera(self):
        for options in [{"camera": -1}, {"width": 0}, {"height": 0},
                        {"process_width": 0}, {"min_size": 0},
                        {"min_neighbors": -1}, {"scale_factor": 1},
                        {"scale_factor": float("nan")}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                fd.video_demo(**options)
        self.camera.assert_not_called()


class CommandLineTests(unittest.TestCase):
    def test_options_reach_camera_loop(self):
        with patch.object(fd, "video_demo") as run:
            status = fd.main(["--camera", "1", "--process-width", "640", "--no-mirror"])
        self.assertEqual(status, 0)
        options = run.call_args.kwargs
        self.assertEqual(options["camera"], 1)
        self.assertEqual(options["process_width"], 640)
        self.assertFalse(options["mirror"])

    def test_invalid_options_are_rejected_before_opening_camera(self):
        invalid = [["--camera", "-1"], ["--width", "0"], ["--height", "-1"],
                   ["--process-width", "0"], ["--min-size", "0"],
                   ["--min-neighbors", "-1"], ["--width", "abc"],
                   ["--scale-factor", "1"], ["--scale-factor", "nan"],
                   ["--scale-factor", "inf"], ["--scale-factor", "abc"]]
        with patch.object(cv2, "VideoCapture") as camera, redirect_stderr(io.StringIO()):
            for args in invalid:
                with self.subTest(args=args), self.assertRaises(SystemExit) as error:
                    fd.main(args)
                self.assertEqual(error.exception.code, 2)
        camera.assert_not_called()

    def test_runtime_failure_returns_nonzero_status(self):
        output = io.StringIO()
        with patch.object(fd, "video_demo", side_effect=RuntimeError("camera unavailable")), \
                redirect_stderr(output):
            self.assertEqual(fd.main([]), 1)
        self.assertIn("camera unavailable", output.getvalue())

    def test_keyboard_interrupt_returns_130(self):
        with patch.object(fd, "video_demo", side_effect=KeyboardInterrupt):
            self.assertEqual(fd.main([]), 130)


if __name__ == "__main__":
    unittest.main()
