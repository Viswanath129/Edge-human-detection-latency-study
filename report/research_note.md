# Research Note: Latency-Accuracy Trade-offs in Edge Detection

## Overview
This research investigates the performance characteristics of YOLOv8 models for real-time human detection on edge devices, focusing on resolution, model size, and precision.

## Findings

### 1. Input Resolution
Reducing input resolution from 640 to 416 significantly improves inference speed (FPS) at the cost of detection accuracy for small or distant objects.
- **640x640:** ~3.15 FPS
- **416x416:** ~5.59 FPS (~77% speedup)

### 2. Model Variants
The nano (n) model is significantly more efficient than the small (s) model, making it the preferred choice for resource-constrained environments.
- **YOLOv8n:** ~114ms latency
- **YOLOv8s:** ~272ms latency (~2.4x slower)

### 3. Precision (FP32 vs FP16)
Half-precision (FP16) inference requires dedicated hardware (NPU/GPU with Tensor Cores). On standard CPU architectures, FP16 can lead to severe performance degradation due to lack of native instruction support.
- **FP32:** Native CPU performance.
- **FP16:** Extremely slow on standard CPU emulated paths.

## Conclusion
For optimal real-time performance on edge CPUs, a combination of **YOLOv8n** at **416x416** resolution with **FP32** precision provides the best balance of speed and usability.
