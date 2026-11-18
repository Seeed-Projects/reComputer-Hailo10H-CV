import threading

import numpy as np

from hailo_platform import HEF, VDevice, FormatType


INFERENCE_TIMEOUT_MS = 10_000


class HailoInfer:
    """Persistent HailoRT 5.1.1 session for YOLO-World.

    YOLO-World is a two-input network: a 640x640 RGB image plus a batch of
    CLIP text embeddings (80 x 512). The image input is fed as float32 and
    HailoRT converts it to the HEF's native UINT8 input; the text input is
    float32 end to end. Both inputs must be supplied to every inference.
    """

    def __init__(self, hef_path, shared_device=None):
        # Pipelines that load two HEFs must share one VDevice: each VDevice()
        # claims the physical device, so a second VDevice() fails with
        # HAILO_OUT_OF_PHYSICAL_DEVICES(74) on single-accelerator boards.
        self.target = shared_device if shared_device is not None else VDevice()
        self._owns_device = shared_device is None
        self._run_lock = threading.Lock()
        self._released = False

        self.hef = HEF(hef_path)
        self.model = self.target.create_infer_model(hef_path)

        self.input_infos = list(self.hef.get_input_vstream_infos())
        self.image_input = None
        self.text_input = None
        for info in self.input_infos:
            shape = tuple(int(v) for v in info.shape)
            if len(shape) >= 3 and shape[-1] == 3 and shape[-2] > 32:
                self.image_input = info
            elif 512 in shape:
                self.text_input = info
        if self.image_input is None:
            raise ValueError(
                "No image input (H, W, 3) found in HEF: "
                + ", ".join(f"{i.name}{tuple(i.shape)}" for i in self.input_infos)
            )

        # YOLO-World needs the text embeddings; fail early instead of running
        # an image-only inference that would silently produce nonsense.
        if self.text_input is None:
            raise ValueError(
                "YOLO-World HEF has no 512-d text-embedding input: "
                + ", ".join(f"{i.name}{tuple(i.shape)}" for i in self.input_infos)
            )

        # Keep each input in its native HEF format: UINT8 for the image and
        # FLOAT32 for the CLIP embeddings.
        self.model.input(self.image_input.name).set_format_type(FormatType.UINT8)
        self.model.input(self.text_input.name).set_format_type(FormatType.FLOAT32)

        image_shape = tuple(int(v) for v in self.image_input.shape)
        if len(image_shape) == 4:
            self.input_h, self.input_w = image_shape[1], image_shape[2]
        else:
            self.input_h, self.input_w = image_shape[0], image_shape[1]

        self._image_shape = image_shape
        self._text_shape = tuple(int(v) for v in self.text_input.shape)
        self._text_embeddings = None

        output_vstream_infos = self.hef.get_output_vstream_infos()
        if not output_vstream_infos:
            raise ValueError("Model has no output vstreams")

        self.output_names = []
        self._output_dtypes = {}
        for info in output_vstream_infos:
            # Always request FLOAT32 outputs: the HEF metadata format is the
            # on-chip quantized encoding, and HailoRT dequantizes when the
            # requested host format is FLOAT32.
            self.model.output(info.name).set_format_type(FormatType.FLOAT32)
            self.output_names.append(info.name)
            self._output_dtypes[info.name] = np.float32

        print(f"Hailo infer model loaded: {hef_path}")
        for info in self.input_infos:
            print(f"  Input:  {info.name} shape={tuple(info.shape)}")
        for name in self.output_names:
            print(f"  Output: {name} shape={self.model.output(name).shape}")

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

    def set_text_embeddings(self, embeddings):
        """Install the vocabulary embeddings used by every inference.

        `embeddings` must contain exactly the number of (class, 512) rows the
        HEF exposes as input_layer2. The bundled COCO file is (1, 80, 512).
        """
        arr = np.ascontiguousarray(embeddings, dtype=np.float32)
        expected = int(np.prod(self._text_shape))
        if arr.size != expected:
            raise ValueError(
                f"Text embedding size {arr.size} != HEF input size {expected} "
                f"(shape {self._text_shape})"
            )
        self._text_embeddings = arr.reshape(self._text_shape)
        print(
            f"Text embeddings installed: shape={self._text_embeddings.shape}, "
            f"rows={self._text_embeddings.size // 512}",
            flush=True,
        )

    @staticmethod
    def _match_shape(array, expected_shape):
        """Add or drop a single batch dimension to match an HEF vstream."""
        expected = tuple(int(v) for v in expected_shape)
        if array.ndim == len(expected):
            return array
        if array.ndim + 1 == len(expected):
            return array.reshape((1,) + array.shape)
        if array.ndim == len(expected) + 1 and array.shape[0] == 1:
            return array.reshape(expected)
        raise ValueError(
            f"Cannot map buffer shape {array.shape} to HEF shape {expected}"
        )

    def run(self, image, text_embeddings=None):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = self._match_shape(image, self._image_shape)

        if text_embeddings is None:
            embeddings = self._text_embeddings
        else:
            embeddings = np.ascontiguousarray(text_embeddings, dtype=np.float32)
            embeddings = self._match_shape(embeddings, self._text_shape)
        if embeddings is None:
            raise RuntimeError(
                "Text embeddings are not installed; call set_text_embeddings()"
            )

        image = np.ascontiguousarray(image)
        embeddings = np.ascontiguousarray(embeddings)

        # Preview, REST and offline analysis can call the same executor from
        # different threads. One persistent binding set is therefore guarded,
        # and outputs are copied before the lock is released so a later frame
        # cannot overwrite data that is still being post-processed.
        with self._run_lock:
            if self._released:
                raise RuntimeError("HailoInfer has already been released")
            self._bindings.input(self.image_input.name).set_buffer(image)
            self._bindings.input(self.text_input.name).set_buffer(embeddings)
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