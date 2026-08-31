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
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.format = _Format()


class _HEF:
    output_specs = [("stdc1/argmax1", [1024, 1920])]

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
    def __init__(self, output_buffers):
        self.input_stream = _BindingStream()
        self.output_streams = {
            name: _BindingStream(buffer)
            for name, buffer in output_buffers.items()
        }

    def input(self):
        return self.input_stream

    def output(self, name=None):
        if name is None:
            return next(iter(self.output_streams.values()))
        return self.output_streams[name]


class _ConfiguredModel:
    def __init__(self):
        self.bindings = None
        self.output_buffers = None
        self.run_bindings = None
        self.run_timeout = None
        self.enter_count = 0
        self.exit_count = 0
        self.create_bindings_count = 0
        self.run_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_count += 1
        return False

    def create_bindings(self, output_buffers=None):
        self.create_bindings_count += 1
        self.output_buffers = output_buffers
        self.bindings = _Bindings(output_buffers)
        return self.bindings

    def run(self, bindings, timeout):
        self.run_count += 1
        self.run_bindings = bindings
        self.run_timeout = timeout
        for buffer in self.output_buffers.values():
            buffer.fill(self.run_count)


class _InferModel:
    def __init__(self):
        self.configured_model = _ConfiguredModel()
        self.configure_count = 0
        self.outputs = {
            name: _ModelInfo(name, shape)
            for name, shape in _HEF.output_specs
        }

    def input(self):
        return _ModelInfo("stdc1/input_layer1", [1024, 1920, 3])

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
        sys.modules["hailo_platform"] = hailo_platform

        spec = importlib.util.spec_from_file_location(
            "stdc1_hailo_executor", EXECUTOR_PATH
        )
        cls.executor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.executor)

    def setUp(self):
        _VDevice.instances.clear()
        _HEF.output_specs = [("stdc1/argmax1", [1024, 1920])]

    def test_configuration_and_bindings_are_reused_across_frames(self):
        infer = self.executor.HailoInfer("model/stdc1.hef")
        image = np.zeros((1024, 1920, 3), dtype=np.uint8)

        first_outputs = infer.run(image)
        second_outputs = infer.run(image)

        model = _VDevice.instances[-1].model
        configured = model.configured_model
        self.assertEqual(model.configure_count, 1)
        self.assertEqual(configured.enter_count, 1)
        self.assertEqual(configured.create_bindings_count, 1)
        self.assertEqual(configured.run_count, 2)
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
        self.assertIsNot(
            first_outputs["stdc1/argmax1"],
            configured.output_buffers["stdc1/argmax1"],
        )
        np.testing.assert_array_equal(first_outputs["stdc1/argmax1"], 1)
        np.testing.assert_array_equal(second_outputs["stdc1/argmax1"], 2)

        infer.release()
        self.assertEqual(configured.exit_count, 1)
        self.assertTrue(_VDevice.instances[-1].released)

    def test_multiple_outputs_use_one_persistent_binding_set(self):
        _HEF.output_specs = [
            ("yolo26/head0", [20, 20, 64]),
            ("yolo26/proto", [160, 160, 32]),
        ]
        infer = self.executor.HailoInfer("model/yolo26n_seg.hef")

        outputs = infer.run(np.zeros((640, 640, 3), dtype=np.uint8))

        configured = _VDevice.instances[-1].model.configured_model
        self.assertEqual(configured.create_bindings_count, 1)
        self.assertEqual(
            set(configured.output_buffers),
            {"yolo26/head0", "yolo26/proto"},
        )
        self.assertEqual(set(outputs), set(configured.output_buffers))
        self.assertTrue(all(value.mean() == 1 for value in outputs.values()))


if __name__ == "__main__":
    unittest.main()
