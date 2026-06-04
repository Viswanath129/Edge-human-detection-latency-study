## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution, model configuration, and numerical precision affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency. Our benchmarking utility supports both live webcam feeds and synthetic fallback for headless environments.

## Experiments
- **Resolution Comparison**: 640x640 vs 416x416
- **Model Size Comparison**: YOLOv8 Nano vs Small
- **Precision Analysis**: FP32 vs FP16

## Benchmark Results (Summary)
| Resolution | Model | Precision | FPS | Avg Latency (ms) | Notes |
|------------|-------|-----------|-----|------------------|-------|
| 416x416    | yolov8n | FP32 | 18.29 | 52.94 | Faster inference |
| 640x640    | yolov8n | FP32 | 8.89 | 108.64 | Balanced performance |
| 640x640    | yolov8n | FP16 | 8.83 | 109.44 | CPU non-optimized |
| 640x640    | yolov8s | FP32 | 4.30 | 228.91 | Higher accuracy |

## Observations
- **Resolution**: Lowering input resolution is the most effective way to improve FPS on edge devices.
- **Model Size**: The Nano variant is essential for real-time performance on constrained hardware.
- **Precision**: FP16 does not provide speedups on standard CPUs without specialized hardware acceleration.

## Future Work
- Edge NPU benchmarking (Coral, Jetson)
- INT8 Quantization tests
- TensorRT integration for Jetson devices
