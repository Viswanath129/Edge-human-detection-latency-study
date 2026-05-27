# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Overview
This research explores the performance characteristics of YOLOv8 variants on edge-constrained environments, focusing on three primary dimensions: input resolution, model architecture size, and numerical precision.

## Experimental Results

### 1. Input Resolution Impact
Higher input resolutions (640x640) provide superior detection for small or distant objects but incur significant latency penalties. Reducing resolution to 416x416 typically yields a ~40-50% reduction in inference time, making it suitable for high-speed tracking at the cost of fine-grained accuracy.

### 2. Model Architecture (Nano vs. Small)
The `yolov8n` (nano) model is highly optimized for edge CPUs. Moving to `yolov8s` (small) increases parameter count, which improves mean Average Precision (mAP) but significantly increases latency. In many edge scenarios, the nano model provides the best balance for real-time performance.

### 3. Precision (FP32 vs. FP16)
FP16 (Half Precision) is designed to leverage hardware accelerators (NVIDIA Tensor Cores, ARM NEON). On standard general-purpose CPUs, FP16 might not show significant speedups and can sometimes be slower due to lack of native hardware support for half-precision arithmetic.

## Conclusion
For real-time human detection on limited hardware, a 416x416 resolution with the `yolov8n` model in FP32 precision currently offers the most stable and responsive performance.
