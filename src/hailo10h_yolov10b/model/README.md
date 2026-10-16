# Model artifact

`yolov10b.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov10b.hef
```

SHA256: `d168a8a3d1dfa8f30f7e06c2553335a8662018f55e57e3008ccc36c2ad176c2c`
Size: 19,619,840 bytes

The compiled HEF contains the on-chip HPP NMS (`yolov8_nms_postprocess_*`
nodes) and exposes a single output vstream (`info.output_shape: 80x5x100`).

Do not replace it with a HEF for another accelerator; HEF files are
hardware-specific.
