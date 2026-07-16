# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Executive Summary
This analysis explores the performance characteristics of YOLOv8 models for real-time human detection on resource-constrained edge devices. We evaluated the impact of input resolution, model architecture, and numerical precision on inference latency and throughput (FPS).

## Experimental Results

### 1. Input Resolution Impact
Reducing the input resolution from 640x640 to 416x416 significantly decreases inference latency.
- **Trade-off:** Lower resolution yields higher FPS but may miss smaller or distant human targets due to reduced spatial information.

### 2. Model Architecture Comparison
We compared the YOLOv8 Nano (n) and Small (s) variants.
- **Nano:** Optimized for maximum speed, suitable for very low-power devices.
- **Small:** Provides a better balance of accuracy and speed if the hardware allows.

### 3. Precision-Aware Inference
FP16 (Half Precision) was evaluated as a method to accelerate inference on compatible hardware (e.g., NVIDIA Jetson, NPU).
- **Observation:** On standard CPUs, FP16 often falls back to FP32 or may even be slower due to lack of specialized instructions. On GPUs, it typically provides a ~2x speedup with minimal accuracy loss.

## Conclusions for Edge Deployment
- For maximum responsiveness, use **YOLOv8n** at **416x416**.
- For high-reliability security applications where humans may be far from the camera, **640x640** resolution is preferred.
- Hardware acceleration (CUDA/TensorRT) with **FP16** is highly recommended for production environments to maintain >30 FPS.

## Hardware Environment
*Benchmarks were conducted in a simulated edge environment (CPU-only fallback where applicable).*
