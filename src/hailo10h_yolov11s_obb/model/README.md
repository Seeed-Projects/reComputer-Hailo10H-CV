# Model artifact

`yolov11s_obb.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov11s_obb.hef
```

SHA256: `a6e8fef423ae805b58a18e77d30d9e4905619a608085f18da7803e15649c7b99`
Size: 17,088,512 bytes

The compiled HEF exposes the nine raw OBB heads (64ch DFL box / 15ch class /
1ch angle per stride 8/16/32) and no on-chip NMS; the host decodes and runs the
rotated probabilistic-IoU NMS (see the module README).

Do not replace it with a HEF for another accelerator; HEF files are
hardware-specific.
