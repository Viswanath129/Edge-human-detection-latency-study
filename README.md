## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices. We evaluate YOLOv8 variants across different resolutions, model sizes, and precision levels.

## Research Questions
1. How does input resolution (640 vs 416) impact inference speed?
2. What is the performance penalty of moving from Nano to Small model variants?
3. How does half-precision (FP16) affect inference on standard CPU-based edge environments?

## Methodology
We use a standardized benchmarking pipeline (`experiments/utils.py`) that handles:
- Automated model downloading and loading.
- Synthetic frame fallback for headless/webcam-less environments.
- 5-frame warmup to stabilize hardware clocks.
- Precise inference timing using `time.perf_counter()`.
- Automated data consolidation into a central summary.

## Experimental Results

### Summary Table
| Resolution | Model | Precision | FPS | Avg Latency (ms) | Observation |
|------------|-------|-----------|-----|------------------|-------------|
| 640x640    | YOLOv8n | FP32 | 9.67 | 103.36 | Baseline (Fastest model) |
| 416x416    | YOLOv8n | FP32 | 21.31 | 46.92 | ~2x speedup |
| 640x640    | YOLOv8n | FP16 | 0.14 | 7221.50 | Extremely slow on CPU |
| 640x640    | YOLOv8s | FP32 | 4.09 | 244.48 | ~2.4x slower than Nano |

### Key Findings
- **Resolution:** Reducing resolution from 640 to 416 provides a nearly linear speedup (~2x FPS), making it the most effective optimization for edge devices without specialized hardware.
- **Model Size:** Moving from YOLOv8n (Nano) to YOLOv8s (Small) increases latency by ~140%, which may be prohibitive for real-time applications on low-power CPUs.
- **Precision:** FP16 inference on standard CPUs (without specific acceleration) is non-viable, resulting in a massive performance degradation. This highlights the importance of hardware-aware optimization (e.g., OpenVINO, TensorRT).

## Visualizations
Generated plots can be found in `results/plots/`:
- `fps_vs_resolution.png`: Performance gains from resolution scaling.
- `latency_vs_resolution.png`: Impact of resolution on per-frame delay.
- `latency_vs_model_size.png`: Scaling from Nano to Small architecture.
- `latency_vs_precision.png`: CPU performance penalty for FP16 (log scale).

## Limitations
- Benchmarks conducted on general-purpose CPU; results will differ on NPU/GPU-equipped edge devices (Jetson, Coral).
- Accuracy is not quantitatively measured in this phase (focus on latency).

## Future Work
- Integration with OpenVINO for CPU acceleration.
- Evaluation on dedicated edge hardware (Jetson Nano, Raspberry Pi 5).
- Quantitative mAP (mean Average Precision) analysis.
