# Model artifact

`nanodet_repvgg.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0:

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/nanodet_repvgg.hef
```

SHA256: `aa3b4796d00b27dcd2eb3b13ba11fa948c4b1429ee01343820ccf3a6b6498b23`
Size: 5,214,208 bytes

The HEF uses on-chip NMS (HPP); the first inference prints the actual input
and output tensor shapes, and those must be treated as the source of truth
for the preprocessing and post-processing code.
