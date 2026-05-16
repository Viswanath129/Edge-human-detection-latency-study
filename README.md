## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution, model size, and precision configuration affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency during inference. The benchmarks support headless environments by falling back to synthetic frames when a camera is unavailable.

## Experiments
- **Resolution comparison**: 640×640 vs 416×416.
- **Model Size comparison**: YOLOv8 Nano vs YOLOv8 Small.
- **Precision comparison**: FP32 vs FP16 precision levels.

## Results Summary
| Category | Parameter | FPS | Avg Latency (ms) | Notes |
|----------|-----------|-----|------------------|-------|
| Resolution | 640x640 | 7.50 | 129.21 | Baseline |
| Resolution | 416x416 | 8.68 | 113.47 | Moderate speedup |
| Model Size | YOLOv8n | 8.93 | 108.22 | Nano variant - optimized |
| Model Size | YOLOv8s | 3.71 | 265.91 | Small variant - more accurate |
| Precision | FP32 | 7.66 | 126.59 | Full precision |
| Precision | FP16 | 0.14 | 7378.17 | Not recommended for CPU |

## Observations
- Lower input resolution improves inference latency, though the gain is moderate on CPU-only environments.
- Model variant choice (Nano vs Small) has a dramatic impact on performance, with Nano being significantly more suitable for real-time edge use.
- FP16 precision results in a massive performance penalty on CPU, likely due to lack of native hardware support and software emulation.

## Limitations
- Synthetic frame fallback for benchmarking in headless environments.
- CPU-only benchmarking (lack of Edge TPU/NPU/GPU tests).
- Approximate accuracy estimation.

## Future Work
- Edge NPU benchmarking (e.g., Coral, Jetson).
- Quantization-aware training (INT8).
- Multi-person tracking performance.
