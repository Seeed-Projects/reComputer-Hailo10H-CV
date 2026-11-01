# Model artifact

`yolov7e6.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov7e6.hef
```

SHA256: `972fbd160c84aab5797e52cd28680653359edd22cfc4b4ae732b463b0f46f15a`
Size: 73,072,640 bytes

The compiled HEF contains the on-chip HPP NMS (`yolov5_nms_postprocess_*`
nodes, `max_proposals_per_class 80`) and exposes a single output vstream
(`info.output_shape: 80x5x80`).

Do not replace it with a HEF for another accelerator; HEF files are
hardware-specific.
