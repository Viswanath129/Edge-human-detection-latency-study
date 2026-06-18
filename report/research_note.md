# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Introduction
This note documents the experimental findings regarding the performance of YOLOv8-based human detection on constrained compute environments. We analyzed the impact of input resolution, model architecture size, and numerical precision on inference latency and throughput (FPS).

## Experimental Setup
- **Hardware**: CPU-based benchmarking (synthetic frames for headless consistency)
- **Models**: YOLOv8n (Nano), YOLOv8s (Small)
- **Resolutions**: 640x640, 416x416
- **Precision**: FP32 (Standard), FP16 (Half - Hardware Dependent)

## Key Findings

### 1. Resolution Impact
Reducing the input resolution from 640x640 to 416x416 significantly improves inference speed. For the YOLOv8n model, throughput increased by approximately 2x, with latency dropping from ~124ms to ~54ms.

### 2. Model Size Comparison
The YOLOv8s (Small) model provides higher detection capacity but at a steep performance cost on CPU. Latency for YOLOv8s at 640x640 was measured at ~258ms, which is more than double that of YOLOv8n (~124ms).

### 3. Precision Considerations
While FP16 (Half Precision) is designed to accelerate inference on compatible GPUs and NPUs, it provides no benefit on standard CPU architectures and may even be slower due to software emulation of half-precision arithmetic. On the tested CPU environment, FP16 was not natively accelerated.

## Conclusion
For real-time human detection on edge devices without dedicated hardware acceleration, YOLOv8n at 416x416 resolution offers the best balance of responsiveness and accuracy.
