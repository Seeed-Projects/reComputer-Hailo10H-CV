# Model artifact

`yolov10n.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov10n.hef
```

SHA256: `babee5194253f5b81f412a358219210113c0f7c9e0c04c15777948311aa00e9e`
Size: 5,230,592 bytes

The compiled HEF contains the on-chip HPP NMS (`yolov8_nms_postprocess_*`
nodes) and exposes a single output vstream (`info.output_shape: 80x5x100`).

Do not replace it with a HEF for another accelerator; HEF files are
hardware-specific.
