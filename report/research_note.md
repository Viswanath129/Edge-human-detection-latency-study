# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Experimental Setup
- **Hardware**: Generic Edge Device Simulation (Synthetic Benchmarking)
- **Model**: YOLOv8 (Nano and Small variants)
- **Framework**: Ultralytics YOLOv8 (PyTorch)
- **Metrics**: Inference Latency (ms) and Throughput (FPS)

## Key Findings

### 1. Impact of Input Resolution
Reducing input resolution is the most effective way to decrease latency without changing the model architecture.

| Resolution | Model | Latency (ms) | FPS |
|------------|-------|--------------|-----|
| 640x640    | Nano  | 106.88       | 9.36|
| 416x416    | Nano  | 49.86        | 20.06|

*Observation*: Dropping resolution to 416x416 resulted in a ~53% reduction in latency.

### 2. Model Complexity vs. Performance
The choice between model variants (Nano vs. Small) significantly impacts the real-time capabilities of the device.

| Model | Resolution | Latency (ms) | FPS |
|-------|------------|--------------|-----|
| Nano  | 640x640    | 106.88       | 9.36|
| Small | 640x640    | 248.54       | 4.02|

*Observation*: The "Small" model is ~2.3x slower than the "Nano" model. On highly constrained edge devices, the Nano model is often the only viable choice for real-time applications (>15 FPS).

### 3. Precision Optimization
FP16 inference was tested; however, on standard CPU environments without specific hardware acceleration (like CUDA or specialized NPUs), FP16 typically falls back to FP32 or does not provide significant speedups. True FP16 acceleration requires compatible GPU hardware.

## Visualizations
The generated plots in `results/plots/` further illustrate these trade-offs:
- `latency_vs_resolution.png`: Highlights the linear-like scaling of latency with pixel count.
- `latency_vs_model.png`: Shows the exponential jump in latency as model depth/width increases.

## Conclusion
For real-time human detection on edge devices, the **YOLOv8n** model at **416x416** resolution provides the best balance, achieving over 20 FPS in our simulated environment. Further optimization via TensorRT or OpenVINO is recommended for production deployment.
