# Research Note: Latency-Accuracy Trade-offs in YOLOv8 for Edge Devices

## Overview
This research explores the impact of input resolution, model size, and numerical precision on the inference performance of YOLOv8-based human detection on edge hardware.

## Key Findings

### 1. Resolution Impact
- **Observation**: Reducing resolution from 640x640 to 416x416 nearly doubles the FPS (from ~8.9 to ~18.3).
- **Trade-off**: Lower resolution reduces the effective receptive field, potentially missing smaller objects, but is critical for maintaining real-time responsiveness on low-power devices.

### 2. Model Scaling
- **Observation**: The `nano` variant (yolov8n) is significantly faster than the `small` variant (yolov8s), achieving ~8.9 FPS vs ~4.3 FPS at 640x640.
- **Trade-off**: `yolov8n` uses ~3.2M parameters compared to `yolov8s`'s ~11.1M. The nano model is preferred for high-throughput edge cases, while small is better suited for stationary high-accuracy monitoring.

### 3. Numerical Precision
- **Observation**: On standard CPU environments, FP16 precision shows no significant performance benefit over FP32 (latencies are near-identical).
- **Recommendation**: FP16/INT8 optimization should be paired with specialized hardware (NVIDIA TensorRT, ARM NEON, or Edge TPUs) to see meaningful acceleration.

## Conclusion
For real-time human detection on resource-constrained edge devices, a combination of **low resolution (416x416)** and **nano-scale models** provides the best balance of speed and detection reliability.
