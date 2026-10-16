# Model artifact

`yolov10s.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov10s.hef
```

SHA256: `1ca968754d1f96c2402462c04fc139ba9ccad9f6c45b0724461592a01006a31b`
Size: 10,182,656 bytes

The compiled HEF contains the on-chip HPP NMS (`yolov8_nms_postprocess_*`
nodes) and exposes a single output vstream (`info.output_shape: 80x5x100`).

Do not replace it with a HEF for another accelerator; HEF files are
hardware-specific.
