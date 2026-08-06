import numpy as np

from hailo_platform import HEF, VDevice


INFERENCE_TIMEOUT_MS = 10_000


class HailoInfer:
    def __init__(self, hef_path):
        self.target = VDevice()

        # HailoRT 5.x new API (Hailo-10H compatible)
        self.hef = HEF(hef_path)
        self.model = self.target.create_infer_model(hef_path)

        self.input_info = self.model.input()
        self.output_info = self.model.output()

        output_vstream_infos = self.hef.get_output_vstream_infos()
        if len(output_vstream_infos) != 1:
            raise ValueError(
                f"Expected one output tensor, got {len(output_vstream_infos)}"
            )

        output_format_type = output_vstream_infos[0].format.type
        output_dtype_name = str(output_format_type).split(".")[-1].lower()
        try:
            self.output_dtype = np.dtype(output_dtype_name)
        except TypeError as exc:
            raise ValueError(
                f"Unsupported Hailo output format: {output_format_type}"
            ) from exc

        # Make the configured output format match the HEF metadata used to
        # allocate the output buffer below.
        self.output_info.set_format_type(output_format_type)

        # Shape may be (H, W, C) or (N, H, W, C) depending on API version
        shape = self.input_info.shape
        if len(shape) == 4:
            self.input_h, self.input_w = shape[1], shape[2]
        else:
            self.input_h, self.input_w = shape[0], shape[1]

        print(f"Hailo infer model loaded: {hef_path}")
        print(f"  Input:  {self.input_info.name} shape={self.input_info.shape}")
        print(f"  Output: {self.output_info.name} shape={self.output_info.shape}")

    def run(self, image):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        with self.model.configure() as configured_model:
            output_buffer = np.empty(
                self.output_info.shape,
                dtype=self.output_dtype,
            )
            bindings = configured_model.create_bindings(
                output_buffers={self.output_info.name: output_buffer}
            )
            bindings.input().set_buffer(image)
            # HailoRT 5.1.1 expects an iterable of Bindings, even for a
            # single-frame inference. The timeout unit is milliseconds.
            configured_model.run([bindings], timeout=INFERENCE_TIMEOUT_MS)
            output = bindings.output(self.output_info.name).get_buffer()

        return {self.output_info.name: output}

    def release(self):
        try:
            self.target.release()
        except Exception:
            pass
