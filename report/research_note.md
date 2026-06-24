# Edge AI Inference: Latency–Accuracy Trade-off Analysis

## Overview
This research investigates the performance impact of various optimization techniques for real-time human detection on edge devices using YOLOv8. We analyze three primary dimensions: input resolution, model architecture size, and inference precision.

## Experimental Results

### 1. Impact of Input Resolution
Reducing input resolution is a highly effective way to decrease latency, albeit with a trade-off in detection granularity for small objects.

| Resolution | Model | Precision | FPS | Latency (ms) |
|------------|-------|-----------|-----|--------------|
| 640x640    | yolov8n | FP32 | 8.53 | 113.39 |
| 416x416    | yolov8n | FP32 | 17.82 | 54.43 |

**Finding:** Dropping resolution from 640 to 416 nearly doubled the FPS and halved the per-frame latency.

### 2. Impact of Model Architecture Size
Comparing the Nano (n) and Small (s) variants of YOLOv8.

| Model | Resolution | Precision | FPS | Latency (ms) |
|-------|------------|-----------|-----|--------------|
| yolov8n | 640x640 | FP32 | 8.53 | 113.39 |
| yolov8s | 640x640 | FP32 | 4.54 | 216.45 |

**Finding:** The Small model is significantly more computationally expensive, resulting in ~50% lower throughput compared to the Nano variant.

### 3. Impact of Precision (Inference-only)
*Note: FP16 benchmarks were conducted with CPU fallback; significant speedups are typically observed only on compatible hardware (e.g., CUDA, NPUs).*

| Precision | Model | Resolution | FPS | Latency (ms) |
|-----------|-------|------------|-----|--------------|
| FP32      | yolov8n | 640x640 | 8.53 | 113.39 |

## Conclusion
For real-time human detection on resource-constrained edge devices:
1. **Resolution scaling** provides the most immediate latency relief.
2. **Nano-sized models** are the most viable for maintainable framerates (e.g., >15 FPS).
3. **Hardware acceleration** is critical for leveraging lower precision (FP16/INT8) benefits.
