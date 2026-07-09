# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Experimental Setup
- **Hardware:** CPU-only environment (simulated edge constraints)
- **Model:** YOLOv8 (Nano and Small variants)
- **Framework:** Ultralytics YOLOv8, PyTorch
- **Methodology:** 50-frame inference benchmark with a 5-frame warmup phase. Synthetic frames used for environment consistency.

## Key Findings

### 1. Impact of Input Resolution
Reducing input resolution significantly improves throughput and reduces latency.

| Resolution | Model | Precision | FPS | Latency (ms) |
|------------|-------|-----------|-----|--------------|
| 640x640    | Nano  | FP32      | 9.0 | 110.9        |
| 416x416    | Nano  | FP32      | 19.1| 52.3         |

**Observation:** Switching from 640x640 to 416x416 nearly doubles the FPS (approx. 112% increase) while halving the latency. This is the most effective optimization for real-time performance on restricted hardware.

### 2. Model Architecture Complexity
The jump from Nano to Small model variants introduces significant computational overhead.

| Model | Resolution | Precision | FPS | Latency (ms) |
|-------|------------|-----------|-----|--------------|
| Nano  | 640x640    | FP32      | 9.0 | 110.9        |
| Small | 640x640    | FP32      | 4.1 | 242.8        |

**Observation:** The YOLOv8s (Small) model is approximately 2.2x slower than YOLOv8n (Nano). For many edge use cases, the minor accuracy gain of the 'Small' model may not justify the significant drop in frame rate below real-time requirements.

### 3. Precision (FP32 vs FP16)
- **Status:** FP16 benchmarks were skipped due to lack of CUDA acceleration.
- **Note:** On standard CPU hardware, FP16 often incurs a performance penalty due to lack of native half-precision optimization in the instruction set, whereas NPU/GPU hardware would show significant speedups.

## Visualizations
Comparative plots can be found in `results/plots/`:
- `latency_vs_resolution.png`: Shows the linear-like scaling of latency with pixel count.
- `latency_vs_model.png`: Illustrates the performance cost of model depth and width.
- `fps_vs_resolution.png`: Highlights the throughput gains at lower resolutions.

## Recommendations for Edge Deployment
1. **Prioritize Resolution Scaling:** Start with 416x416 for human detection; it offers the best balance for real-time tracking.
2. **Nano-first Approach:** Use YOLOv8n as the baseline. Only move to larger models if specific classes or small-object detection requirements are not met.
3. **Hardware Acceleration:** Deploy on hardware supporting FP16/INT8 (e.g., NVIDIA Jetson, Coral Edge TPU) to unlock further latency reductions.
