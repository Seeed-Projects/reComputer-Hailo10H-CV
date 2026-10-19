# Model artifact

`yolov11n_obb.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov11n_obb.hef
```

SHA256: `6188275241433279adc067c9a419c216aa1bc7212e0baf1297731015112f1008`
Size: 6,893,568 bytes

The compiled HEF exposes the nine raw OBB heads (64ch DFL box / 15ch class /
1ch angle per stride 8/16/32) and no on-chip NMS; the host decodes and runs the
rotated probabilistic-IoU NMS (see the module README).

Do not replace it with a HEF for another accelerator; HEF files are
hardware-specific.
