# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Experimental Setup
- **Hardware**: CPU-based benchmarking (Headless Environment)
- **Model Architecture**: Ultralytics YOLOv8 (Nano vs. Small)
- **Input Resolutions**: 640x640, 416x416
- **Precision**: FP32 (FP16 tested but fallback to FP32 due to lack of CUDA acceleration)

## Key Findings

### 1. Impact of Input Resolution
Reducing the input resolution from 640x640 to 416x416 resulted in a significant performance boost.
- **Latency**: Reduced by ~55% (from 116.57ms to 52.09ms).
- **Throughput**: Increased from ~8.3 FPS to ~18.6 FPS.

Resolution scaling remains the most effective lever for real-time performance on resource-constrained hardware.

### 2. Model Architecture Comparison (Nano vs. Small)
The transition from YOLOv8n (Nano) to YOLOv8s (Small) provides better feature extraction at a high computational cost.
- **YOLOv8n**: ~8.3 FPS
- **YOLOv8s**: ~3.8 FPS (Approx. 54% slower than Nano)

For edge devices without NPU/GPU acceleration, the Nano variant is the only viable option for near-real-time applications.

### 3. Precision (FP32 vs FP16)
In this environment, FP16 inference was not accelerated due to the absence of CUDA-capable hardware. On standard CPUs, FP32 remains the most stable and performant precision level.

## Visualizations
The following plots (located in `results/plots/`) illustrate these trade-offs:
- `fps_vs_resolution.png`: Shows the linear gain in throughput with resolution reduction.
- `latency_vs_resolution.png`: Highlights the latency savings.
- `fps_vs_model.png`: Compares throughput across different model scales.

## Conclusion
For real-time human detection on the edge, a combination of **YOLOv8n** and **416x416 resolution** offers the best balance, achieving >15 FPS on standard CPU hardware. Further optimization would require hardware acceleration (NPU/GPU) and potential quantization (INT8).
