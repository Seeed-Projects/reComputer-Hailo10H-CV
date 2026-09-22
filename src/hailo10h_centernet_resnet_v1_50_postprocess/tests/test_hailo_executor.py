import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


EXECUTOR_PATH = Path(__file__).parents[1] / "py_utils" / "hailo_executor.py"

# Mock output specs shaped like the CenterNet HEF vstreams:
#   centernet0_conv3          -> wh  (128x128x2)
#   centernet0_conv5          -> reg (128x128x2)
#   threshold_confidence/...  -> hm  (128x128x80)
DEFAULT_OUTPUT_SPECS = [
    ("centernet0_conv3", [128, 128, 2]),
    ("centernet0_conv5", [128, 128, 2]),
    ("threshold_confidence/conv29", [128, 128, 80]),
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
        return _ModelInfo("centernet_resnet_v1_50_postprocess/input_layer1", [512, 512, 3])

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

        spec = importlib.util.spec_from_file_location(
            "centernet_hailo_executor", EXECUTOR_PATH
        )
        cls.executor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.executor)

    def setUp(self):
        _VDevice.instances.clear()
        _HEF.output_specs = [("threshold_confidence/conv29", [128, 128, 80])]

    def test_configuration_and_bindings_are_reused_across_frames(self):
        infer = self.executor.HailoInfer(
            "model/centernet_resnet_v1_50_postprocess.hef"
        )
        image = np.zeros((512, 512, 3), dtype=np.uint8)

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
            (1, 512, 512, 3),
        )
        self.assertEqual(
            set(configured.output_buffers), {"threshold_confidence/conv29"}
        )
        self.assertEqual(
            configured.output_buffers["threshold_confidence/conv29"].shape,
            (128, 128, 80),
        )
        # HailoRT 5.x dequantizes on read: the executor always requests
        # FLOAT32 output buffers, never the raw on-chip quantized encoding.
        self.assertEqual(
            configured.output_buffers["threshold_confidence/conv29"].dtype,
            np.dtype("float32"),
        )
        self.assertIsNot(
            first_outputs["threshold_confidence/conv29"],
            configured.output_buffers["threshold_confidence/conv29"],
        )
        np.testing.assert_array_equal(
            first_outputs["threshold_confidence/conv29"], 1
        )
        np.testing.assert_array_equal(
            second_outputs["threshold_confidence/conv29"], 2
        )

        infer.release()
        self.assertEqual(configured.exit_count, 1)
        self.assertTrue(_VDevice.instances[-1].released)

    def test_three_centernet_heads_use_one_persistent_binding_set(self):
        _HEF.output_specs = list(DEFAULT_OUTPUT_SPECS)
        infer = self.executor.HailoInfer(
            "model/centernet_resnet_v1_50_postprocess.hef"
        )

        outputs = infer.run(np.zeros((512, 512, 3), dtype=np.uint8))

        configured = _VDevice.instances[-1].model.configured_model
        self.assertEqual(configured.create_bindings_count, 1)
        self.assertEqual(
            set(configured.output_buffers),
            {"centernet0_conv3", "centernet0_conv5", "threshold_confidence/conv29"},
        )
        self.assertEqual(set(outputs), set(configured.output_buffers))
        self.assertTrue(all(value.mean() == 1 for value in outputs.values()))


if __name__ == "__main__":
    unittest.main()