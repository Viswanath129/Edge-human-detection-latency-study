# Research Note: YOLO-based Human Detection on Edge Devices

## Abstract
This study investigates the performance trade-offs between input resolution, model architecture, and precision for real-time human detection tasks. Using YOLOv8 variants, we benchmarked inference latency and throughput (FPS) to provide deployment guidelines for compute-constrained edge environments.

## Experimental Setup
- **Hardware:** CPU-based benchmarking (simulated edge environment)
- **Framework:** Ultralytics YOLOv8
- **Models:** YOLOv8n (Nano), YOLOv8s (Small)
- **Resolutions:** 640x640, 416x416
- **Precision:** FP32 (Standard), FP16 (Half - hardware dependent)

## Key Findings

### 1. Impact of Input Resolution
Reducing input resolution from 640x640 to 416x416 yielded a significant performance boost.
- **Latency reduction:** ~53% (100.6ms down to 47.0ms for YOLOv8n)
- **Throughput increase:** ~114% (9.9 FPS up to 21.3 FPS)
- **Trade-off:** Lower resolutions may reduce detection accuracy for small or distant objects, but are essential for reaching real-time targets (>20 FPS) on limited hardware.

### 2. Model Architecture Scaling
The transition from Nano (yolov8n) to Small (yolov8s) doubles the latency.
- **YOLOv8n Latency:** 100.6ms (at 640x640)
- **YOLOv8s Latency:** 217.3ms (at 640x640)
- **Conclusion:** For edge devices without dedicated NPUs, YOLOv8n is the only viable candidate for near-real-time applications.

### 3. Precision Optimization
FP16 precision was tested, but significant speedups were not observed on CPU-only environments. Actual FP16 acceleration requires hardware support (e.g., NVIDIA Tensor Cores or specific ARM NEON instructions).

## Conclusion
For optimal real-time performance on edge CPUs, we recommend using **YOLOv8n** at **416x416** resolution. This configuration achieved ~21 FPS in our tests, which is sufficient for most human monitoring and security applications.
