# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Experimental Setup
- **Hardware**: CPU-based benchmarking (Synthetic fallback for headless environments)
- **Model**: YOLOv8 (Nano and Small variants)
- **Metric**: Inference-only latency and FPS (averaged over 50 frames, 5-frame warmup)

## Key Findings

### 1. Resolution Impact
Lowering the input resolution from 640x640 to 416x416 significantly reduces latency.
- **640x640**: ~102ms latency (~10 FPS)
- **416x416**: ~46ms latency (~21 FPS)
- **Insight**: 416x416 provides a ~2.2x speedup, making it more suitable for real-time applications on constrained hardware.

### 2. Model Complexity
Comparing YOLOv8n (Nano) vs YOLOv8s (Small) at 640x640 resolution.
- **YOLOv8n**: ~102ms latency
- **YOLOv8s**: ~231ms latency
- **Insight**: The Small model is more than 2x slower than the Nano model. For most edge human detection tasks, the Nano model offers a better performance-latency balance.

### 3. Precision (FP32 vs FP16)
- **Observation**: On standard CPU hardware without specialized optimization, FP16 does not provide a speedup and may even be slower or unsupported. CUDA-accelerated environments are required to realize the benefits of FP16.

## Conclusion
For real-time human detection on edge CPUs, using **YOLOv8n at 416x416 resolution** is the most viable configuration, achieving over 20 FPS.
