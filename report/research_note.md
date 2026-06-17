# Edge Human Detection: Latency-Accuracy Trade-off Analysis

This note summarizes the performance impact of resolution, precision, and model architecture on YOLOv8 inference for human detection on edge-simulated hardware.

## 1. Resolution Impact
Scaling the input resolution from 640x640 to 416x416 significantly impacts throughput.

| Resolution | Model | FPS | Latency (ms) |
|------------|-------|-----|--------------|
| 640x640    | Nano  | 8.3 | 119.9        |
| 416x416    | Nano  | 19.2| 52.2         |

**Analysis:** Reducing resolution by ~35% per dimension resulted in a ~2.3x increase in FPS. This is the most effective lever for achieving real-time performance on restricted hardware, though it may impact detection range for small objects.

## 2. Model Size Comparison
We compared the YOLOv8 Nano (n) and Small (s) variants at 640x640 resolution.

| Model | FPS | Latency (ms) | Params |
|-------|-----|--------------|--------|
| Nano  | 8.3 | 119.9        | 3.2M   |
| Small | 3.2 | 310.4        | 11.2M  |

**Analysis:** The Small model is nearly 3x slower than the Nano model. While it offers higher mAP, the latency penalty (310ms) makes it unsuitable for high-frequency real-time applications on CPU-bound edge devices.

## 3. Precision (FP16 vs FP32)
Tests were conducted at 640x640 resolution on CPU.

| Precision | FPS | Latency (ms) |
|-----------|-----|--------------|
| FP32      | 8.4 | 118.5        |
| FP16      | 8.7 | 115.6        |

**Analysis:** On standard CPU hardware without specific FP16 acceleration (like NPU or Tensor Core), the performance gains from half-precision are negligible (<3%). Significant speedups are expected when deployed on hardware with native FP16 support.

## Conclusion
For maximum throughput on edge devices, a combination of **416x416 resolution** and **Nano architecture** is recommended, delivering ~19 FPS. If hardware acceleration (GPU/NPU) is available, enabling **FP16** should be prioritized to further reduce latency without additional architectural changes.
