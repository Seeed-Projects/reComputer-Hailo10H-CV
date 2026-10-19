# Model artifact

`yolov11m_obb.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov11m_obb.hef
```

SHA256: `91647602a7d0edceed1f51101b8af704b27e8fc3ab8e8a2686d141c1b7c620dc`
Size: 30,142,464 bytes

The compiled HEF exposes the nine raw OBB heads (64ch DFL box / 15ch class /
1ch angle per stride 8/16/32) and no on-chip NMS; the host decodes and runs the
rotated probabilistic-IoU NMS (see the module README).

Do not replace it with a HEF for another accelerator; HEF files are
hardware-specific.
