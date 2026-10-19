# Model artifact

`yolov11x_obb.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov11x_obb.hef
```

SHA256: `f9f6a3b2c563c7bd2aa337955802b47b31d8aaa26f6ec0e02cd59b03282cdeac`
Size: 64,696,320 bytes

The compiled HEF exposes the nine raw OBB heads (64ch DFL box / 15ch class /
1ch angle per stride 8/16/32) and no on-chip NMS; the host decodes and runs the
rotated probabilistic-IoU NMS (see the module README).

Do not replace it with a HEF for another accelerator; HEF files are
hardware-specific.
