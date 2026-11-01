# Model artifact

`yolov7_tiny.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov7_tiny.hef
```

SHA256: `4e419ac94202833cf98a3bec59ff84aa63c636e3d4de26f88b4388d9133f2324`
Size: 9,199,616 bytes

The compiled HEF contains the on-chip HPP NMS (`yolov5_nms_postprocess_*`
nodes, `max_proposals_per_class 80`) and exposes a single output vstream
(`info.output_shape: 80x5x80`).

Do not replace it with a HEF for another accelerator; HEF files are
hardware-specific.
