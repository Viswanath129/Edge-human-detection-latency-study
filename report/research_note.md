# Research Note: Edge Inference Trade-offs for Human Detection

## Executive Summary
This research explores the performance characteristics of YOLOv8-based human detection on edge compute environments. We analyze three primary dimensions: input resolution, model architecture (parameters), and numerical precision.

## Experimental Results

### 1. Impact of Input Resolution
Reducing input resolution from 640x640 to 416x416 provides a significant boost in throughput.
- **Latency Reduction:** ~40-50% decrease in per-frame inference time.
- **Throughput Increase:** Nearly double the FPS.
- **Trade-off:** Lower resolution reduces the effective receptive field, making small object detection (e.g., humans at a distance) more challenging.

### 2. Model Architecture: Nano vs Small
Switching from YOLOv8n (Nano) to YOLOv8s (Small) increases model capacity but impacts real-time performance.
- **Nano:** Optimized for extreme edge devices; provides the highest FPS.
- **Small:** Offers better feature extraction and localization at the cost of higher computational latency.

### 3. Numerical Precision: FP32 vs FP16
Half-precision (FP16) inference is critical for hardware-accelerated edge devices (e.g., NVIDIA Jetson, NPU).
- **FP32:** Standard precision; baseline for accuracy.
- **FP16:** When supported by hardware (CUDA/TensorRT), it can significantly reduce memory bandwidth usage and improve latency without substantial accuracy loss. *Note: On standard CPUs, FP16 may actually be slower due to lack of native hardware acceleration.*

## Key Recommendations
1. **For Real-Time Monitoring:** Prioritize 416x416 resolution with YOLOv8n to maintain >15 FPS.
2. **For High-Accuracy Analytics:** Use 640x640 with YOLOv8s, ideally leveraging FP16 acceleration if a compatible NPU/GPU is available.
3. **Adaptive Resolution:** Future implementations could dynamically adjust resolution based on the detected human density or battery constraints.
