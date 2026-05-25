# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Introduction
Real-time human detection on edge devices is constrained by limited compute power. This note analyzes the impact of three primary variables on inference performance: input resolution, model architecture size, and numerical precision.

## Analysis

### 1. Input Resolution Scaling
Reducing the input resolution from 640x640 to 416x416 significantly reduces the number of operations required per forward pass. Our benchmarks show a reduction in latency from ~101ms to ~49ms.
- **Impact:** Significant improvement in FPS, suitable for high-motion scenarios.
- **Trade-off:** Potential loss in small-object detection accuracy.

### 2. Model Architecture Depth
Comparing YOLOv8n (3.2M parameters) to YOLOv8s (11.2M parameters).
- **Latency Impact:** Increase from 101ms to 265ms.
- **Inference Density:** The Small model is considerably more "heavyweight," requiring a shift from frame-by-frame processing to potential skipping or aggressive downsampling on edge CPUs.

### 3. Numerical Precision (FP32 vs FP16)
Half-precision (FP16) is designed for modern GPUs and specialized AI accelerators (NPUs).
- **Observation:** On general-purpose CPUs used in this test, FP16 is not natively accelerated and often runs via software emulation or inefficient casting, leading to a ~70x increase in latency.
- **Conclusion:** FP16 should only be deployed where hardware support is verified.

## Engineering Recommendations
1. **Prioritize Resolution Tuning:** Scaling resolution is the "low-hanging fruit" for FPS optimization.
2. **Nano-first Approach:** Always start with the smallest model variant for edge deployment.
3. **Hardware-Specific Compilation:** Use toolkits like OpenVINO or ONNX Runtime to leverage hardware-specific optimizations if FP16/INT8 is required.
