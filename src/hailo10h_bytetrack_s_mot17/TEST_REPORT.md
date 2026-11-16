# Test Report - ByteTrack-S (MOT17) on CM5 + Hailo-10H

## Model

| Field | Value |
|---|---|
| Model name | bytetrack_s_mot17 |
| Task | MOT17 pedestrian detection + ByteTrack association |
| Parameters | 8.94M |
| Operations | 43.21G |
| MOTA (float / Hailo-10H) | 79.5 / 78.1 |
| Model Zoo version | v5.5.0 |
| Compile target | hailo10h |

## HEF artifact

| Field | Value |
|---|---|
| File | model/bytetrack_s_mot17.hef |
| Size | 14,815,232 bytes |
| SHA256 | `9c872ee8d22bc718a3d82794ec0c03a7cd6997d784cfbd8761e0ee8063947623` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/bytetrack_s_mot17.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 608x1088x3 RGB uint8 (letterbox, padding 114) |
| On-chip NMS | NO (9 raw heads) |
| Output | 3 strides x (box 4, obj 1, class 1): 19x34, 38x68, 76x136 |
| Classes | 1 (person) |
| Decode | YOLOX-style center/size decode, score=obj*class, NMS 0.7 |
| Tracking | ByteTrack (Kalman + IoU matching), host-side |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Offline nine-head decode + tracker ID persistence | Pass |
| Configuration defaults (score 0.1, NMS 0.7, track 0.5) | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |

### Hardware verification - pending

| Check | Status |
|---|---|
| HEF loads, output head shapes confirmed | Pending |
| Demo video: person boxes and stable track IDs | Pending |
| Single-image REST API | Pending |
| Official GHCR image re-pull + run | Pending |
