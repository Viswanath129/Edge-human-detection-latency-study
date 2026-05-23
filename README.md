## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices, focusing on YOLOv8 variants.

## Research Question
How do input resolution, model configuration, and inference precision affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency. The benchmarking suite supports:
- **Frame Warmup**: 5-frame warmup to stabilize hardware/software states.
- **Inference-only FPS**: Metrics are calculated strictly based on inference latency to avoid I/O noise.
- **Headless Support**: Support for `FORCE_SYNTHETIC=true` to run benchmarks without a physical webcam.

## Experiments
- **Resolution Comparison**: 640×640 vs 416×416 (YOLOv8n).
- **Precision-aware Inference**: FP32 vs FP16 comparison.
- **Model Size Comparison**: YOLOv8n (Nano) vs YOLOv8s (Small).

## Benchmark Results
| Resolution | Model | Precision | FPS | Avg Latency (ms) | Notes |
|------------|-------|-----------|-----|------------------|-------|
| 640x640    | yolov8n | FP32      | 9.48 | 105.51           | Base benchmark |
| 416x416    | yolov8n | FP32      | 9.17 | 108.99           | Lower resolution |
| 640x640    | yolov8n | FP16      | 0.14 | 7393.63          | Slow on CPU (no NPU) |
| 640x640    | yolov8s | FP32      | 4.29 | 233.16           | Larger model |

## Observations
- Lower input resolution typically improves latency, though benefits vary by hardware.
- FP16 precision is significantly slower on standard CPUs without hardware acceleration (NPU/CUDA).
- Moving from Nano to Small model variants roughly doubles inference latency.

## Limitations
- Benchmarking performed in a simulated environment (synthetic frames).
- FP16 performance is representative only for specialized edge hardware.
- Single-device benchmarking.

## Future Work
- Edge NPU benchmarking (Jetson Nano / Coral TPU).
- INT8 quantization analysis.
- Multi-object detection trade-offs.
