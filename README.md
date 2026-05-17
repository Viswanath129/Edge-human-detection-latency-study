## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution, model size, and precision affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLOv8-based human detection pipeline and benchmark performance across three dimensions:
1. **Resolution:** 640x640 vs 416x416
2. **Model Variant:** Nano (n) vs Small (s)
3. **Precision:** FP32 vs FP16

The benchmarking scripts support headless environments via synthetic frame fallback if no camera is detected.

## Benchmark Results
| Category | Variant | FPS | Avg Latency (ms) | Notes |
|----------|---------|-----|------------------|-------|
| Resolution | 640x640 | 3.15 | 203.77 | Base configuration |
| Resolution | 416x416 | 5.59 | 117.49 | 1.7x FPS gain |
| Model Size | YOLOv8n | 5.59 | 113.83 | Recommended for edge |
| Model Size | YOLOv8s | 2.49 | 271.92 | High accuracy, low FPS |
| Precision | FP32 | 2.68 | 131.81 | Standard |
| Precision | FP16 | 0.05 | 7218.25 | CPU bottleneck |

## Observations
- Lower input resolution significantly improves inference latency.
- YOLOv8n (Nano) is essential for achieving near real-time performance on edge CPUs.
- FP16 precision is only beneficial on hardware with native half-precision support (CUDA/NPU).

## Limitations
- Performance highly dependent on host CPU architecture.
- Accuracy measured qualitatively via inference confidence thresholds.
- Single-threaded benchmarking (potential for multi-stream optimization).

## Future Work
- Edge NPU benchmarking (Coral, Hailo)
- TensorRT / OpenVINO optimization
- Multi-object tracking latency analysis
