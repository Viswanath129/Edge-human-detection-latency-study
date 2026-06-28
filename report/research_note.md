# Research Note: Latency-Accuracy Trade-offs for Edge Human Detection

## Experimental Setup
We evaluated YOLOv8 variants on a simulated edge environment using synthetic frames to ensure consistent benchmarking. The primary metrics are inference-only Latency (ms) and Frames Per Second (FPS).

## Key Findings

### 1. Impact of Input Resolution
Reducing input resolution from 640x640 to 416x416 significantly improves performance.
- **Latency Reduction:** ~50% decrease (from ~115ms to ~56ms).
- **Throughput Increase:** ~2x improvement in FPS (from ~8.7 to ~17.7).
- **Trade-off:** Lower resolution reduces the model's ability to detect small objects or people at a distance.

### 2. Model Architecture Scaling
Switching from YOLOv8n (Nano) to YOLOv8s (Small) increases computational demand.
- **Latency Increase:** ~2.3x higher latency (from ~115ms to ~266ms).
- **FPS Drop:** Significant drop from ~8.7 to ~3.75 FPS.
- **Trade-off:** YOLOv8s provides better feature extraction and higher mAP, but may not be suitable for real-time applications on low-power edge devices without acceleration.

### 3. Precision Optimization
FP16 precision was evaluated as a potential optimization.
- **Hardware Requirement:** FP16 requires CUDA-enabled hardware for meaningful performance gains.
- **Observation:** On standard CPU hardware, FP32 remains the representative precision, as FP16 emulation is often slower.

## Conclusion
For real-time human detection on edge devices, **YOLOv8n at 416x416 resolution** offers the best balance between speed and detection capability, achieving nearly 18 FPS in a single-thread inference loop.
