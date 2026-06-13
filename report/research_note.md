# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Executive Summary
This analysis explores the performance characteristics of YOLOv8 variants on edge-constrained environments, focusing on the impact of input resolution and model complexity on inference latency and throughput.

## Key Findings

### 1. Impact of Input Resolution
Reducing the input resolution from **640x640** to **416x416** resulted in a significant performance boost.
- **Latency Reduction:** ~50% decrease in per-frame inference time.
- **Throughput Increase:** FPS nearly doubled, enabling smoother real-time processing.
- **Trade-off:** Lower resolution reduces the effective range and sensitivity for small object detection, which is critical for human detection in large-scale environments.

### 2. Model Variant Comparison (Nano vs. Small)
Switching from the **Nano (yolov8n)** variant to the **Small (yolov8s)** variant highlights the cost of increased model depth and width.
- **Nano Performance:** ~10 FPS (on current test hardware).
- **Small Performance:** ~4 FPS.
- **Insight:** The Nano variant is significantly better suited for real-time applications on low-power edge devices, while the Small variant may require hardware acceleration (GPU/NPU) to achieve acceptable frame rates.

### 3. Precision Considerations
- **FP32 vs FP16:** In CPU-only environments, FP16 typically falls back to FP32 or runs slower via software emulation.
- **Future Direction:** Deploying on hardware with native FP16/INT8 support (like Jetson Orin or specialized NPUs) is expected to further reduce latency by 1.5x - 3x.

## Recommendations
- Use **YOLOv8n** with **416x416** resolution for maximum responsiveness on standard edge CPUs.
- If high accuracy for distant subjects is required, use **640x640** resolution with **YOLOv8n** and consider a lightweight hardware accelerator.
