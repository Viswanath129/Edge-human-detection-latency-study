# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Overview
This study evaluates the performance of YOLOv8 variants on simulated edge compute environments, focusing on three key dimensions: input resolution, precision, and model size.

## Key Findings

### 1. Resolution Impact
Reducing input resolution from 640x640 to 416x416 provides the most significant performance boost for real-time applications.
- **FPS Improvement:** ~2.3x increase (9.11 FPS → 20.97 FPS).
- **Latency Reduction:** ~56.6% decrease.
- **Trade-off:** Lower resolution reduces the detection range and accuracy for small objects.

### 2. Precision Trade-offs (FP32 vs FP16)
Benchmarking FP16 on CPU-only environments shows a massive performance regression.
- **Observation:** FP16 latency is ~65x higher than FP32 on standard CPUs.
- **Conclusion:** FP16 is strictly intended for hardware-accelerated environments (CUDA/NPU). Without specialized hardware support, standard FP32 remains the optimal choice for CPU-based edge inference.

### 3. Model Size Scaling
Scaling from YOLOv8n (Nano) to YOLOv8s (Small) doubles the latency.
- **Nano (FP32, 640px):** ~9.11 FPS / 110ms latency.
- **Small (FP32, 640px):** ~4.50 FPS / 222ms latency.
- **Insight:** YOLOv8n is the only viable candidate for real-time CPU-only inference on low-power edge devices, especially when combined with 416px resolution.

## Recommended Configuration for Edge Devices
For real-time human detection on limited hardware, we recommend:
- **Model:** YOLOv8n
- **Resolution:** 416x416
- **Precision:** FP32 (unless NPU/GPU acceleration is available)
