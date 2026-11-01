# Model artifact

`yolov7.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov7.hef
```

SHA256: `eb8cc72c8df91ea2865b02e2cd970791276806fa58bd5fc0e4cc9846d3a52cef`
Size: 28,106,752 bytes

The compiled HEF contains the on-chip HPP NMS (`yolov5_nms_postprocess_*`
nodes, `max_proposals_per_class 80`) and exposes a single output vstream
(`info.output_shape: 80x5x80`).

Do not replace it with a HEF for another accelerator; HEF files are
hardware-specific.
