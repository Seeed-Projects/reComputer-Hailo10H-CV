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


class _BindingStream:
    def __init__(self, buffer=None):
        self.buffer = buffer

    def set_buffer(self, buffer):
        self.buffer = buffer

    def get_buffer(self):
        return self.buffer


class _Bindings:
    def __init__(self):
        self.input_stream = _BindingStream()
        self.output_stream = _BindingStream(
            np.zeros((1, 1024, 1920), dtype=np.uint8)
        )

    def input(self):
        return self.input_stream

    def output(self):
        return self.output_stream


class _ConfiguredModel:
    def __init__(self):
        self.bindings = _Bindings()
        self.run_bindings = None
        self.run_timeout = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def create_bindings(self):
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
        hailo_platform.VDevice = _VDevice
        sys.modules["hailo_platform"] = hailo_platform

        spec = importlib.util.spec_from_file_location(
            "stdc1_hailo_executor", EXECUTOR_PATH
        )
        cls.executor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.executor)

    def setUp(self):
        _VDevice.instances.clear()

    def test_single_binding_is_wrapped_and_timeout_is_milliseconds(self):
        infer = self.executor.HailoInfer("model/stdc1.hef")
        image = np.zeros((1024, 1920, 3), dtype=np.uint8)

        outputs = infer.run(image)

        configured = _VDevice.instances[-1].model.configured_model
        self.assertEqual(configured.run_bindings, [configured.bindings])
        self.assertEqual(configured.run_timeout, 10_000)
        self.assertEqual(configured.bindings.input_stream.buffer.shape, (1, 1024, 1920, 3))
        self.assertIn("stdc1/argmax1", outputs)


if __name__ == "__main__":
    unittest.main()
