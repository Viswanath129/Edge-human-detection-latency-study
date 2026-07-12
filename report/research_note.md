# Research Note: Latency-Accuracy Trade-offs for YOLOv8 on Edge Devices

## Overview
This study explores how input resolution, model architecture, and precision affect inference performance for real-time human detection.

## Experimental Setup
- **Hardware**: Generic CPU (benchmarked in a containerized environment)
- **Model**: YOLOv8 (Nano and Small variants)
- **Tooling**: Ultralytics API, OpenCV, PyTorch
- **Metrics**: Average Latency (ms), Throughput (FPS)

## Methodology
Inference is performed on synthetic frames to simulate real-time video processing. Each model undergoes a 5-frame warmup to stabilize performance metrics. Latency is measured strictly for the inference call (and GPU synchronization if applicable).

## Results Summary

| Resolution | Model | Precision | Average FPS | Avg Latency (ms) | Observation |
|------------|-------|-----------|-------------|------------------|-------------|
| 640x640    | YOLOv8n | FP32      | ~8.7        | ~115.0           | Standard baseline |
| 416x416    | YOLOv8n | FP32      | ~19.6       | ~51.0            | 2.2x speedup via resolution scaling |
| 640x640    | YOLOv8s | FP32      | ~3.9        | ~256.7           | Higher capacity, significantly slower |

## Key Findings
1. **Resolution Impact**: Reducing input resolution from 640 to 416 provided a ~55% reduction in latency for the YOLOv8n model.
2. **Model Scaling**: Moving from YOLOv8n to YOLOv8s more than doubled the latency, indicating a steep performance cost for increased model complexity on edge CPU hardware.
3. **Hardware Constraints**: Performance on CPU remains the primary bottleneck for real-time applications requiring >30 FPS at high resolutions.

## Future Work
- Integration of INT8 quantization for edge deployment.
- Testing on specialized hardware (NVIDIA Jetson, Edge TPUs).
- Accuracy benchmarking using the COCO dataset.
