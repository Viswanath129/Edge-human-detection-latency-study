# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Experimental Setup
- **Model Architecture:** YOLOv8 (Nano and Small variants)
- **Input Resolutions:** 640x640, 416x416
- **Precision:** FP32 (Full Precision)
- **Hardware:** Benchmarked using pre-generated synthetic frames to ensure inference-only performance measurement.

## Key Findings

### 1. Impact of Input Resolution
Reducing the input resolution from 640x640 to 416x416 resulted in a significant performance boost.
- **Latency:** Decreased from ~98.74ms to ~47.51ms.
- **Throughput:** Increased from ~10.13 FPS to ~21.05 FPS.
- **Trade-off:** Lower resolution improves real-time performance at the cost of detection accuracy for small or distant objects.

### 2. Impact of Model Complexity
Comparing YOLOv8n (Nano) and YOLOv8s (Small) at 640x640 resolution:
- **Nano:** ~10.13 FPS | ~98.74ms Latency
- **Small:** ~4.48 FPS | ~223.21ms Latency
- **Trade-off:** The Small model is approximately 2.2x slower than the Nano model, reflecting the increased depth and parameter count. It is suitable for applications where high detection precision is mandatory.

### 3. Precision Considerations
Initial tests focused on FP32. While FP16 was considered, lack of hardware-specific acceleration (CUDA) on the test environment prevented a representative comparison. On compatible edge hardware, FP16 is expected to provide significant throughput gains.

## Conclusion
For real-time human detection on resource-constrained edge devices, **YOLOv8n at 416x416** provides the most viable balance, achieving >20 FPS. If higher accuracy is required, **YOLOv8n at 640x640** (~10 FPS) offers a solid middle ground before stepping up to larger model variants.
