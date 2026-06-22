# Latency–Accuracy Trade-offs in Real-Time Human Detection

## Methodology
In this study, we evaluated the performance of YOLOv8 models (nano and small variants) on edge-representative hardware. We focused on three key dimensions:
1. **Input Resolution:** Comparing 640x640 vs 416x416.
2. **Model Architecture:** Comparing YOLOv8n (nano) vs YOLOv8s (small).
3. **Precision:** Evaluating the impact of FP16 (Half Precision) vs FP32 (Full Precision).

Performance was measured in Frames Per Second (FPS) and Average Latency (ms) over a benchmark of 50 frames, following a 5-frame warmup phase.

## Experimental Results

| Resolution | Model | Precision | Average FPS | Avg Latency (ms) | Notes |
|------------|-------|-----------|-------------|------------------|-------|
| 640x640    | Nano  | FP32      | ~8.8        | ~114             | Baseline |
| 416x416    | Nano  | FP32      | ~19.2       | ~52              | ~2.2x speedup via resolution reduction |
| 640x640    | Small | FP32      | ~3.9        | ~255             | ~2.2x slower than Nano for better accuracy |

*(Note: FP16 results were not collected in the current CPU-only environment but are architecturally designed to provide ~1.5-2x speedups on supported GPU hardware.)*

## Analysis of Trade-offs

### Resolution Scaling
Reducing the input resolution from 640 to 416 yielded a significant performance boost (~2.2x increase in FPS). This is the most impactful optimization for extremely constrained edge devices, though it may reduce the detection range for small objects.

### Model Architecture
The transition from YOLOv8n to YOLOv8s roughly doubles the latency. While YOLOv8s offers higher mean Average Precision (mAP), the trade-off in real-time responsiveness is substantial, potentially dropping below the 5-10 FPS threshold required for smooth tracking in some applications.

### Hardware Acceleration
The benchmarking suite is ready for GPU-enabled edge devices (e.g., Jetson Nano/Orin). On such devices, FP16 precision and NPU/GPU acceleration are critical to achieving >30 FPS at 640x640 resolution.

## Conclusion
For real-time human detection on CPU-based edge devices, **YOLOv8n at 416x416 resolution** provides the most viable balance, achieving near 20 FPS. If higher accuracy is required, upgrading to YOLOv8s is recommended only if hardware acceleration (GPU/NPU) is available.
