# Model artifacts

This pipeline bundles two Hailo-10H HEFs from Hailo Model Zoo v5.4.0:

| File | Source | Size | SHA256 |
|---|---|---:|---|
| `lprnet.hef` | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/lprnet.hef | 4,665,344 | `e1109d07dc7a2854cc23d901e3d3a7c1b7d4e49238fbe351a9c9c0acd08b73b8` |
| `tiny_yolov4_license_plates.hef` | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/tiny_yolov4_license_plates.hef | 5,095,424 | `1f84a8cc065362bc7382d95601710601c3826bcb2cf6de2e3716944a1624c2c5` |

Note: an earlier build accidentally bundled the Hailo-8 detector HEF
(8.2 MB, sha 97ef98...) alongside the Hailo-10H lprnet — it failed on
device with HAILO_HEF_NOT_COMPATIBLE_WITH_DEVICE(93). Do not replace
either file with a Hailo-8/8L or Hailo-15H HEF; HEF files are
hardware-specific.
