import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


EXECUTOR_PATH = Path(__file__).parents[1] / "py_utils" / "hailo_executor.py"


class _ModelInfo:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.format_type = None

    def set_format_type(self, format_type):
        self.format_type = format_type


class _FormatType:
    def __str__(self):
        return "FormatType.UINT8"


class _Format:
    def __init__(self):
        self.type = _FormatType()


class _OutputVStreamInfo:
    def __init__(self):
        self.name = "stdc1/argmax1"
        self.format = _Format()


class _HEF:
    def __init__(self, hef_path):
        self.hef_path = hef_path

    def get_output_vstream_infos(self):
        return [_OutputVStreamInfo()]


class _BindingStream:
    def __init__(self, buffer=None):
        self.buffer = buffer

    def set_buffer(self, buffer):
        self.buffer = buffer

    def get_buffer(self):
        return self.buffer


class _Bindings:
    def __init__(self, output_buffers):
        self.input_stream = _BindingStream()
        self.output_stream = _BindingStream(output_buffers["stdc1/argmax1"])

    def input(self):
        return self.input_stream

    def output(self, name=None):
        return self.output_stream


class _ConfiguredModel:
    def __init__(self):
        self.bindings = None
        self.output_buffers = None
        self.run_bindings = None
        self.run_timeout = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def create_bindings(self, output_buffers=None):
        self.output_buffers = output_buffers
        self.bindings = _Bindings(output_buffers)
        return self.bindings

    def run(self, bindings, timeout):
        self.run_bindings = bindings
        self.run_timeout = timeout


class _InferModel:
    def __init__(self):
        self.configured_model = _ConfiguredModel()

    def input(self):
        return _ModelInfo("stdc1/input_layer1", [1024, 1920, 3])

    def output(self):
        return _ModelInfo("stdc1/argmax1", [1024, 1920])

    def configure(self):
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
        sys.modules["hailo_platform"] = hailo_platform

        spec = importlib.util.spec_from_file_location(
            "stdc1_hailo_executor", EXECUTOR_PATH
        )
        cls.executor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.executor)

    def setUp(self):
        _VDevice.instances.clear()

    def test_output_buffer_is_registered_before_single_frame_run(self):
        infer = self.executor.HailoInfer("model/stdc1.hef")
        image = np.zeros((1024, 1920, 3), dtype=np.uint8)

        outputs = infer.run(image)

        configured = _VDevice.instances[-1].model.configured_model
        self.assertEqual(configured.run_bindings, [configured.bindings])
        self.assertEqual(configured.run_timeout, 10_000)
        self.assertEqual(
            configured.bindings.input_stream.buffer.shape,
            (1, 1024, 1920, 3),
        )
        self.assertEqual(set(configured.output_buffers), {"stdc1/argmax1"})
        self.assertEqual(
            configured.output_buffers["stdc1/argmax1"].shape,
            (1024, 1920),
        )
        self.assertEqual(
            configured.output_buffers["stdc1/argmax1"].dtype,
            np.dtype("uint8"),
        )
        self.assertIs(
            outputs["stdc1/argmax1"],
            configured.output_buffers["stdc1/argmax1"],
        )


if __name__ == "__main__":
    unittest.main()
