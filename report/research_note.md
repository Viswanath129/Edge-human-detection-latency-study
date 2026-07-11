# Research Note: Latency-Accuracy Trade-offs in YOLO-based Human Detection

## Experimental Setup
- **Model:** YOLOv8 (Nano vs. Small)
- **Resolution:** 640x640 and 416x416
- **Environment:** CPU-based inference benchmarking using synthetic frames to eliminate I/O noise.

## Key Findings

### 1. Impact of Input Resolution
Reducing the input resolution from 640x640 to 416x416 resulted in a significant performance boost.
- **Latency Reduction:** ~52% reduction in per-frame inference time.
- **Throughput Increase:** FPS improved from ~9.7 to ~20.5 (on tested CPU).
- **Trade-off:** Lower resolutions may reduce detection accuracy for small or distant objects, but provide substantial gains for real-time edge applications.

### 2. Impact of Model Architecture
Comparing YOLOv8n (Nano) and YOLOv8s (Small) at 640x640:
- **Nano (yolov8n):** ~103ms latency, ~9.7 FPS.
- **Small (yolov8s):** ~240ms latency, ~4.2 FPS.
- **Observation:** The Small model is over 2x slower than the Nano model. While it offers higher precision due to more parameters, the latency cost is prohibitive for many resource-constrained edge devices.

### 3. Precision (FP32 vs. FP16)
- **Current Observation:** Benchmarking on CPU showed that FP16 is not natively accelerated and often falls back to FP32 or runs slower.
- **Hardware Requirement:** For significant latency gains from FP16, hardware acceleration (e.g., NVIDIA Jetson with CUDA, or specialized NPUs) is required.

## Conclusion
For real-time human detection on edge devices, **YOLOv8n at 416x416** provides the best balance of speed and usability, achieving over 20 FPS on standard mobile-grade CPUs. Upgrading to a Small model or higher resolution should only be considered if the hardware includes dedicated acceleration or if the detection of small objects is a critical requirement.
