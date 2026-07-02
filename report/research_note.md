# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Experimental Setup
- **Hardware**: Sandbox Environment (Headless)
- **Model**: YOLOv8 (Nano, Small)
- **Inference Library**: Ultralytics YOLOv8
- **Metrics**:
  - Inference Latency (ms)
  - Frames Per Second (FPS)

## Findings

### 1. Impact of Input Resolution
Reducing input resolution from 640x640 to 416x416 significantly decreases inference latency.
- **640x640**: ~103ms latency (~9.3 FPS)
- **416x416**: ~48ms latency (~19.7 FPS)
- **Insight**: A 35% reduction in pixel count (from 640 to 416 width) results in a ~50% reduction in latency on CPU.

### 2. Model Architecture Comparison
The choice between YOLOv8n (Nano) and YOLOv8s (Small) shows a clear trade-off between speed and complexity.
- **YOLOv8n**: ~9.3 FPS
- **YOLOv8s**: ~4.18 FPS
- **Insight**: Moving from Nano to Small more than doubles the latency, which might be prohibitive for real-time applications on low-power edge devices.

### 3. Precision (FP32 vs FP16)
- On standard CPU hardware without specialized instruction sets or NPU acceleration, FP16 performance is often non-representative.
- Full performance gains from FP16 require CUDA-capable GPUs or specific Edge NPUs.

## Conclusion
For real-time human detection on resource-constrained edge devices, YOLOv8n at 416x416 resolution provides the most viable balance, achieving nearly 20 FPS on CPU-based inference. Further optimizations should explore INT8 quantization and hardware-specific acceleration (TensorRT/OpenVINO).
