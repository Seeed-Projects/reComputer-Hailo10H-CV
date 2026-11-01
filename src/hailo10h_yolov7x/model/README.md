# Model artifact

`yolov7x.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov7x.hef
```

SHA256: `0e2b001da8871de4950168969f2fc3170136575b8cf861e84ec300560b7059b4`
Size: 56,406,016 bytes

The compiled HEF contains the on-chip HPP NMS (`yolov5_nms_postprocess_*`
nodes, `max_proposals_per_class 80`) and exposes a single output vstream
(`info.output_shape: 80x5x80`).

Do not replace it with a HEF for another accelerator; HEF files are
hardware-specific.
