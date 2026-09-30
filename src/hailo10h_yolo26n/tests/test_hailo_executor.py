"""Regression tests for the HailoRT 5.1.1 wrapper (no device required)."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


EXECUTOR_PATH = Path(__file__).parents[1] / "py_utils" / "hailo_executor.py"

# Mock output spec: the wrapper test only exercises binding reuse; the module
# logs the real vstream layout at first inference. Shapes mirror the HEF:
# box head, class scores and the 160x160x32 mask prototype.
DEFAULT_OUTPUT_SPECS = [
    ("yolo26n/conv102", [20, 20, 4]),
    ("yolo26n/conv105", [20, 20, 80]),
]


class _ModelInfo:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.format_type = None

    def set_format_type(self, format_type):
        self.format_type = format_type


class _FormatType:
    FLOAT32 = "FLOAT32"

    def __str__(self):
        return "FormatType.UINT8"


class _Format:
    def __init__(self):
        self.type = _FormatType()


class _OutputVStreamInfo:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.format = _Format()


class _HEF:
    output_specs = list(DEFAULT_OUTPUT_SPECS)

    def __init__(self, hef_path):
        self.hef_path = hef_path

    def get_output_vstream_infos(self):
        return [_OutputVStreamInfo(name, shape) for name, shape in self.output_specs]


class _BindingStream:
    def __init__(self, buffer=None):
        self.buffer = buffer

    def set_buffer(self, buffer):
        self.buffer = buffer

    def get_buffer(self):
        return self.buffer


class _Bindings:
    def __init__(self, input_buffer, output_buffers):
        self.input_stream = _BindingStream(input_buffer)
        self.output_streams = [(_BindingStream(buf), name)
                               for name, buf in output_buffers.items()]

    def input(self):
        return self.input_stream

    def output(self, name=None):
        for stream, stream_name in self.output_streams:
            if name is None or stream_name == name:
                return stream
        raise KeyError(name)


class _ConfiguredModel:
    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0
        self.create_bindings_count = 0
        self.run_count = 0
        self.run_bindings = None
        self.run_timeout = None
        self.bindings = None
        self.output_buffers = None

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, *exc):
        self.exit_count += 1
        return False

    def create_bindings(self, output_buffers=None):
        self.create_bindings_count += 1
        self.output_buffers = dict(output_buffers)
        self.bindings = _Bindings(np.zeros((1, 640, 640, 3), dtype=np.uint8), output_buffers)
        return self.bindings

    def run(self, bindings, timeout=None):
        self.run_count += 1
        self.run_bindings = bindings
        self.run_timeout = timeout
        for stream, _name in self.bindings.output_streams:
            stream.buffer[...] = self.run_count


class _InferModel:
    def __init__(self):
        self.configured_model = _ConfiguredModel()
        self.configure_count = 0
        self.outputs = {name: _ModelInfo(name, shape) for name, shape in _HEF.output_specs}

    def input(self):
        return _ModelInfo("yolo26n/input_layer1", [640, 640, 3])

    def output(self, name=None):
        if name is None:
            return next(iter(self.outputs.values()))
        return self.outputs[name]

    def configure(self):
        self.configure_count += 1
        return self.configured_model


class _VDevice:
    instances = []

    def __init__(self):
        self.model = _InferModel()
        self.released = False
        self.instances.append(self)

    def create_infer_model(self, hef_path):
        self.hef_path = hef_path
        return self.model

    def release(self):
        self.released = True


class HailoExecutorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        hailo_platform = types.ModuleType("hailo_platform")
        hailo_platform.HEF = _HEF
        hailo_platform.VDevice = _VDevice
        hailo_platform.FormatType = _FormatType
        sys.modules["hailo_platform"] = hailo_platform

        spec = importlib.util.spec_from_file_location("yolo26n_hailo_executor", EXECUTOR_PATH)
        cls.executor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.executor)

    def setUp(self):
        _VDevice.instances.clear()
        _HEF.output_specs = list(DEFAULT_OUTPUT_SPECS)

    def test_configuration_and_bindings_are_reused_across_frames(self):
        infer = self.executor.HailoInfer("model/yolo26n.hef")
        image = np.zeros((640, 640, 3), dtype=np.uint8)

        first = infer.run(image)
        second = infer.run(image)

        model = _VDevice.instances[-1].model
        configured = model.configured_model
        self.assertEqual(model.configure_count, 1)
        self.assertEqual(configured.enter_count, 1)
        self.assertEqual(configured.create_bindings_count, 1)
        self.assertEqual(configured.run_count, 2)
        self.assertEqual(configured.run_bindings, [configured.bindings])
        self.assertEqual(configured.run_timeout, 10_000)
        self.assertEqual(configured.bindings.input_stream.buffer.shape, (1, 640, 640, 3))
        names = {name for name, _shape in DEFAULT_OUTPUT_SPECS}
        self.assertEqual(set(configured.output_buffers), names)
        for name, shape in DEFAULT_OUTPUT_SPECS:
            self.assertEqual(configured.output_buffers[name].shape, tuple(shape))
            self.assertEqual(configured.output_buffers[name].dtype, np.dtype("float32"))
        box_name = DEFAULT_OUTPUT_SPECS[0][0]
        self.assertIsNot(first[box_name], configured.output_buffers[box_name])
        np.testing.assert_array_equal(first[box_name], 1)
        np.testing.assert_array_equal(second[box_name], 2)

        infer.release()
        self.assertEqual(configured.exit_count, 1)
        self.assertTrue(_VDevice.instances[-1].released)


if __name__ == "__main__":
    unittest.main()
