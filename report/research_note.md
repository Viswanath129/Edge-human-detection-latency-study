# Research Note: YOLO-based Human Detection Performance on Edge Devices

## Experimental Setup
- **Hardware**: Generic CPU environment (synthetic benchmark)
- **Model Architecture**: Ultralytics YOLOv8 (Nano and Small variants)
- **Input Resolutions**: 640x640 and 416x416
- **Precision**: FP32 (Standard) and FP16 (Half-precision)

## Key Findings

### 1. Impact of Input Resolution
Reducing the input resolution from 640x640 to 416x416 significantly improves inference throughput.
- **YOLOv8n (640x640)**: ~9.7 FPS / 103ms latency
- **YOLOv8n (416x416)**: ~21.1 FPS / 47.5ms latency
- **Performance Gain**: ~117% increase in FPS by reducing resolution.

### 2. Model Size Trade-offs
The Nano model provides a substantial performance advantage over the Small model on resource-constrained hardware.
- **YOLOv8n (Nano)**: ~9.7 FPS / 103ms latency
- **YOLOv8s (Small)**: ~4.4 FPS / 226ms latency
- **Efficiency**: The Nano model is approximately 2.2x faster than the Small variant.

### 3. Precision (FP16 vs FP32)
In CPU-only environments without dedicated hardware acceleration (CUDA/NPU), FP16 inference often falls back to FP32 or does not provide meaningful acceleration. On supported hardware, FP16 is expected to reduce memory footprint and potentially improve throughput.

## Conclusion
For real-time human detection on edge devices with limited compute, **YOLOv8n at 416x416 resolution** offers the best balance between responsiveness and detection capability, achieving over 20 FPS in our tests. Larger models like YOLOv8s may require hardware acceleration (NPUs/GPUs) to reach interactive frame rates.
