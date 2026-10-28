"""Regenerate the tiny integration fixture with: pip install onnx; python this_file.py.

This network always emits one box. It is a protocol fixture, NOT a trained detector.
ONNX is a fixture-generation dependency only, not needed by the test suite.
"""

from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

initializers = [
    numpy_helper.from_array(np.zeros((5, 3, 1, 1), np.float32), "weights"),
    numpy_helper.from_array(np.array([320, 320, 40, 40, 0.95], np.float32), "bias"),
    numpy_helper.from_array(np.array([1, 5, 1], np.int64), "shape"),
]
graph = helper.make_graph(
    [helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
     helper.make_node("Conv", ["pooled", "weights", "bias"], ["boxes"], kernel_shape=[1, 1]),
     helper.make_node("Reshape", ["boxes", "shape"], ["output0"])],
    "synthetic_yolov8_protocol_fixture",
    [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
    [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 1])], initializers,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
onnx.checker.check_model(model)
onnx.save(model, Path(__file__).with_name("synthetic_yolov8.onnx"))
