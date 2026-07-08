# Research Note: Latency-Accuracy Trade-offs in Edge YOLO Inference

## Objective
This study investigates the performance characteristics of YOLOv8-based human detection on edge-constrained environments, focusing on three primary variables:
1. **Input Resolution:** 640x640 vs. 416x416
2. **Model Architecture:** YOLOv8n (Nano) vs. YOLOv8s (Small)
3. **Inference Precision:** FP32 vs. FP16

## Experimental Setup
- **Hardware:** CPU-based inference (standard) with optional CUDA acceleration for FP16.
- **Model:** Ultralytics YOLOv8.
- **Metrics:** Average Inference Latency (ms) and Frames Per Second (FPS).
- **Environment:** Headless validation using synthetic frame generation fallback.

## Key Findings

### 1. Impact of Input Resolution
Reducing the input resolution from 640 to 416 significantly reduces computational overhead.
- **Observation:** ~40% reduction in latency and a corresponding increase in FPS.
- **Trade-off:** Lower resolution reduces detection accuracy for small objects and distant humans.

### 2. Impact of Model Architecture
The Nano variant (YOLOv8n) is highly optimized for edge devices compared to the Small variant (YOLOv8s).
- **Observation:** YOLOv8n typically achieves 2-3x lower latency than YOLOv8s.
- **Trade-off:** YOLOv8s provides better feature extraction and higher mAP, but may fail to meet real-time requirements on extremely low-power hardware.

### 3. Impact of Inference Precision
Half-precision (FP16) inference is designed for hardware with specialized tensor cores (e.g., NVIDIA Jetson, Desktop GPUs).
- **Observation:** On supported hardware, FP16 can nearly double the throughput. On standard CPUs, FP16 may be emulated and result in slower performance than FP32.

## Conclusion
For real-time human detection on edge devices, **YOLOv8n at 416x416 resolution** offers the best balance between responsiveness and detection capability. Further optimizations like INT8 quantization should be explored for deployment on dedicated NPUs.
