# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Experimental Setup
- **Hardware:** CPU-based benchmarking (synthetic frames)
- **Model:** YOLOv8 (Nano and Small variants)
- **Input Resolutions:** 640x640, 416x416
- **Precision:** FP32 (Standard)

## Key Findings

### 1. Impact of Input Resolution
Reducing the input resolution from 640x640 to 416x416 resulted in a significant performance boost.
- **Latency reduction:** ~55% (from ~112ms to ~51ms)
- **FPS improvement:** ~118% (from ~9 FPS to ~19.5 FPS)
- **Trade-off:** Lower resolution reduces the model's ability to detect small objects and distant humans, but provides much smoother real-time performance on constrained hardware.

### 2. Model Size Comparison
Switching from YOLOv8n (Nano) to YOLOv8s (Small) at 640x640 resolution:
- **Latency increase:** ~120% (from ~112ms to ~245ms)
- **FPS drop:** ~55% (from ~9 FPS to ~4 FPS)
- **Trade-off:** The Small model offers higher detection capacity and better accuracy for complex scenes, but its computational cost makes it less suitable for real-time applications on standard CPUs without acceleration.

### 3. Precision (FP32 vs FP16)
- On the tested CPU environment, FP16 did not provide a performance advantage due to lack of native hardware acceleration for half-precision floating point operations.
- For edge devices with specialized hardware (e.g., NVIDIA Jetson with CUDA, or NPUs), FP16 is expected to significantly reduce latency and memory bandwidth usage.

## Conclusion
For real-time human detection on CPU-constrained edge devices, using **YOLOv8n** at **416x416** resolution provides the best balance, achieving nearly 20 FPS. If higher accuracy is required, maintaining 640x640 resolution with the Nano model is preferable over switching to a larger model size unless hardware acceleration (GPU/NPU) is available.
