import threading

import numpy as np

from hailo_platform import HEF, VDevice, FormatType


INFERENCE_TIMEOUT_MS = 10_000


class HailoInfer:
    """Persistent HailoRT 5.1.1 inference session.

    HailoRT model configuration is intentionally kept alive for the lifetime
    of this wrapper. Configuring the model and rebuilding bindings per frame
    adds enough host overhead to collapse real-time throughput.
    """

    def __init__(self, hef_path, shared_device=None):
        # Pipelines that load two HEFs must share one VDevice: each VDevice()
        # claims the physical device, so a second VDevice() fails with
        # HAILO_OUT_OF_PHYSICAL_DEVICES(74) on single-accelerator boards.
        # Pass shared_device=first_model.target when constructing the second.
        self.target = shared_device if shared_device is not None else VDevice()
        self._owns_device = shared_device is None
        self._run_lock = threading.Lock()
        self._released = False

        # HailoRT 5.x new API (Hailo-10H compatible)
        self.hef = HEF(hef_path)
        self.model = self.target.create_infer_model(hef_path)

        self.input_info = self.model.input()

        output_vstream_infos = self.hef.get_output_vstream_infos()
        if not output_vstream_infos:
            raise ValueError("Model has no output vstreams")

        self.output_names = []
        self._output_dtypes = {}
        for info in output_vstream_infos:
            # Always request FLOAT32 outputs: the HEF metadata format is the
            # on-chip quantized encoding (e.g. UINT16 fixed-point), and the
            # HailoRT SDK performs the dequantization when converting to
            # FLOAT32. Reading the raw quantized buffers produced values in
            # the tens of thousands and zero valid detections.
            self.model.output(info.name).set_format_type(FormatType.FLOAT32)
            self.output_names.append(info.name)
            self._output_dtypes[info.name] = np.float32

        # Shape may be (H, W, C) or (N, H, W, C) depending on API version
        shape = self.input_info.shape
        if len(shape) == 4:
            self.input_h, self.input_w = shape[1], shape[2]
        else:
            self.input_h, self.input_w = shape[0], shape[1]

        print(f"Hailo infer model loaded: {hef_path}")
        print(f"  Input:  {self.input_info.name} shape={self.input_info.shape}")
        for name in self.output_names:
            print(f"  Output: {name} shape={self.model.output(name).shape}")

        # HailoRT 5.1.1 InferModel API: enter the configuration context once,
        # then reuse the configured model, bindings and device-side pipeline.
        self._configured_context = self.model.configure()
        self._configured_model = self._configured_context.__enter__()
        self._output_buffers = {
            name: np.empty(
                self.model.output(name).shape,
                dtype=self._output_dtypes[name],
            )
            for name in self.output_names
        }
        self._bindings = self._configured_model.create_bindings(
            output_buffers=self._output_buffers
        )

    def run(self, image):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        # Preview, REST and offline analysis can call the same executor from
        # different threads. One persistent binding set is therefore guarded,
        # and outputs are copied before the lock is released so a later frame
        # cannot overwrite data that is still being post-processed.
        with self._run_lock:
            if self._released:
                raise RuntimeError("HailoInfer has already been released")
            self._bindings.input().set_buffer(image)
            self._configured_model.run(
                [self._bindings], timeout=INFERENCE_TIMEOUT_MS
            )
            return {
                name: self._bindings.output(name).get_buffer().copy()
                for name in self.output_names
            }

    def release(self):
        with self._run_lock:
            if self._released:
                return
            try:
                self._configured_context.__exit__(None, None, None)
            finally:
                if self._owns_device:
                    self.target.release()
                self._released = True
