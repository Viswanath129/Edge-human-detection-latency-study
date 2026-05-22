## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution, model configuration, and precision affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency. Our benchmarking utility supports both live video and synthetic frame generation for testing in headless environments.

## Experiments
- **Resolution comparison**: 640×640 vs 416×416 (YOLOv8n)
- **Model size comparison**: YOLOv8n (Nano) vs YOLOv8s (Small)
- **Precision comparison**: FP32 vs FP16

## Results Summary
Current benchmarks (Run on CPU, Synthetic Frames):

| Model | Resolution | Precision | FPS | Avg Latency (ms) | Observation |
|-------|------------|-----------|-----|------------------|-------------|
| YOLOv8n | 640x640 | FP32 | 8.36 | 115.75 | Lightweight (nano) |
| YOLOv8n | 416x416 | FP32 | 18.19 | 53.25 | Faster Inference |
| YOLOv8s | 640x640 | FP32 | 3.83 | 257.11 | Medium (small) |
| YOLOv8n | 640x640 | FP16 | 0.14 | 7303.98 | Slower on CPU (emulated) |

## Observations
- **Resolution**: Lowering input resolution to 416x416 provides a ~2x speedup in FPS.
- **Model Size**: YOLOv8s is significantly slower than YOLOv8n, offering a trade-off for potentially higher accuracy.
- **Precision**: FP16 inference on standard CPUs is extremely slow as it is not hardware-accelerated. This should be tested on a GPU or NPU for representative results.

## Limitations
- Benchmarks currently run on CPU.
- Limited dataset for accuracy validation.
- FP16 performance is non-representative on this environment.

## Future Work
- Edge NPU benchmarking (TensorRT / OpenVINO)
- Formal mAP accuracy evaluation
- Power consumption analysis
