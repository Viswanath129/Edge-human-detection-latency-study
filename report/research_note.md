# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Overview
This document summarizes experimental findings regarding the impact of input resolution, model architecture, and inference precision on human detection performance on edge-constrained environments.

## Experimental Setup
- **Hardware:** Benchmarked on standard CPU (synthetic frames used for consistency in headless environments).
- **Model:** YOLOv8 (Nano and Small variants).
- **Task:** Real-time human detection.

## Key Findings

### 1. Impact of Input Resolution
Reducing input resolution from 640px to 416px significantly reduces inference latency.
- **640x640 (YOLOv8n):** ~100ms latency (~10 FPS)
- **416x416 (YOLOv8n):** ~45ms latency (~22 FPS)
- **Observation:** A ~55% reduction in resolution leads to a >2x increase in throughput (FPS).

### 2. Impact of Model Architecture
Switching from YOLOv8 Nano (n) to YOLOv8 Small (s) improves accuracy (qualitative) but at a high computational cost.
- **YOLOv8n (640x640):** ~100ms latency
- **YOLOv8s (640x640):** ~210ms latency
- **Observation:** The Small model is ~2x slower than the Nano model at the same resolution.

### 3. Impact of Inference Precision
FP16 (Half-precision) benchmarking was attempted. On standard CPU hardware, FP16 often shows no improvement or even degradation unless specific hardware acceleration (CUDA/NPU) is available.
- In our current CPU-based environment, FP32 remains the standard.
- Future work should target NPU-specific optimizations.

## Conclusion
For real-time edge applications where <50ms latency is required, **YOLOv8n at 416x416** resolution offers the most viable trade-off between speed and detection capability.
