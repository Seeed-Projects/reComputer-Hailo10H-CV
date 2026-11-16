# Test Report - ByteTrack-Tiny (MOT17) on CM5 + Hailo-10H

## Model

| Field | Value |
|---|---|
| Model name | bytetrack_tiny_mot17 |
| Task | MOT17 pedestrian detection + ByteTrack association |
| Parameters | 5.03M |
| Operations | 24.60G |
| MOTA (float / Hailo-10H) | 77.1 / 75.8 |
| Model Zoo version | v5.5.0 |
| Compile target | hailo10h |

## HEF artifact

| Field | Value |
|---|---|
| File | model/bytetrack_tiny_mot17.hef |
| Size | 9,912,320 bytes |
| SHA256 | `4081e5021abe96129721cc778beddb7c1236aa657e1f84e82f9f5cbd1f54cb29` |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/bytetrack_tiny_mot17.hef |

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
