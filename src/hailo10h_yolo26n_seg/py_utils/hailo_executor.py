import numpy as np

from hailo_platform import HEF, VDevice


INFERENCE_TIMEOUT_MS = 10_000


class HailoInfer:
    """Multi-output HailoRT 5.x executor (Hailo-10H compatible).

    Supports any number of output vstreams: single-output models get the
    familiar `{name: array}` dict, multi-output models (e.g. yolo26_seg with
    10 heads) get one buffer per output name.
    """

    def __init__(self, hef_path):
        self.target = VDevice()

        # HailoRT 5.x new API (Hailo-10H compatible)
        self.hef = HEF(hef_path)
        self.model = self.target.create_infer_model(hef_path)

        self.input_info = self.model.input()

        # Enumerate ALL output vstreams (single- or multi-output models).
        output_vstream_infos = self.hef.get_output_vstream_infos()
        if not output_vstream_infos:
            raise ValueError("Model has no output vstreams")

        self.output_names = []
        self._output_dtypes = {}
        for info in output_vstream_infos:
            output_format_type = info.format.type
            output_dtype_name = str(output_format_type).split(".")[-1].lower()
            try:
                dtype = np.dtype(output_dtype_name)
            except TypeError as exc:
                raise ValueError(
                    f"Unsupported Hailo output format: {output_format_type}"
                ) from exc
            # Make the configured output format match the HEF metadata used
            # to allocate the output buffer at run time.
            self.model.output(info.name).set_format_type(output_format_type)
            self.output_names.append(info.name)
            self._output_dtypes[info.name] = dtype

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

    def run(self, image):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        with self.model.configure() as configured_model:
            output_buffers = {
                name: np.empty(
                    self.model.output(name).shape,
                    dtype=self._output_dtypes[name],
                )
                for name in self.output_names
            }
            bindings = configured_model.create_bindings(
                output_buffers=output_buffers
            )
            bindings.input().set_buffer(image)
            # HailoRT 5.1.1 expects an iterable of Bindings, even for a
            # single-frame inference. The timeout unit is milliseconds.
            configured_model.run([bindings], timeout=INFERENCE_TIMEOUT_MS)
            outputs = {
                name: bindings.output(name).get_buffer()
                for name in self.output_names
            }

        return outputs

    def release(self):
        try:
            self.target.release()
        except Exception:
            pass