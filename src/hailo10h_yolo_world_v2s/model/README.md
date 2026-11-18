# Model artifacts

## `yolo_world_v2s.hef` (Hailo-10H)

Published in the official Hailo Model Zoo v5.4.0 `hailo10h` prefix:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolo_world_v2s.hef
```

- Size: `25,092,096` bytes
- SHA256: `489a366562d617ea8428958405e84f37e65775eb4b66e1af112617d1321ef1f0`
- Network: YOLO-World v2s, zero-shot object detection, 640x640 image input,
  80 x 512 CLIP text-embedding input, host-side NMS post-process
- Model Zoo metadata: 9.3M parameters, 32.7G operations, 36.3 COCO mAP
  (full precision), 34.9 hardware mAP, 42.8 FPS batch 1 on Hailo-10H
  (official v5.5.0 benchmark)

The v5.5.0 build is also published and is the one linked from Hailo's public
Hailo-10H zero-shot table. This module uses v5.4.0 to stay aligned with the
other Hailo-10H modules in this repository, which target HailoRT 5.1.1. The
HEF format is v3 and its metadata reports SDK 5.4.0. HEF/runtime
compatibility must still be checked on the board.

## `coco_text_embeddings.npy`

- Size: `163,968` bytes
- SHA256: `ffeb1fe385ac7cc0beefbdc8dede0eec4384f9212e1a193ed4fcdced3a3281be`
- Shape: `(1, 80, 512)`, float32, L2-normalized per row

The 80 rows are CLIP ViT-B/32 text embeddings for the COCO class names in
`coco_classes.txt`, in the same order. They were generated with
`openai/clip-vit-base-patch32` (`CLIPTextModelWithProjection`, class name only,
no prompt template), matching
`hailo_model_zoo/datasets/create_coco_tfrecord_with_text_embeddings.py`.

## `coco_classes.txt`

- Size: `701` bytes
- SHA256: `4d4aaea7bee6be2f675d9b53a9195ca36dfe6429f7479f29155da522a6c85930`

One COCO class name per line, 80 rows. Row `i` labels output class `i` and must
remain aligned with row `i` of the embedding file.

Do not replace the HEF with a Hailo-8, Hailo-15H, or Hailo-8L build; HEF files
are hardware-specific. If you export a custom 80-prompt vocabulary, replace
both the embeddings and the class file together.