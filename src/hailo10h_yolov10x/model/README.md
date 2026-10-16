# Model artifact

`yolov10x.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov10x.hef
```

SHA256: `cb3d6bec9840e936217b2b67cd5147c168f7a3845f932dd3dd9fbf3d1ac7d76d`
Size: 26,251,264 bytes

The compiled HEF contains the on-chip HPP NMS (`yolov8_nms_postprocess_*`
nodes) and exposes a single output vstream (`info.output_shape: 80x5x100`).

Do not replace it with a HEF for another accelerator; HEF files are
hardware-specific.
