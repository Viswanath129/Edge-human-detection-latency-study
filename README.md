## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution, model configuration, and precision levels affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline using Ultralytics YOLOv8. We benchmark performance across three key dimensions:
- **Resolution**: 640x640 vs 416x416
- **Model Variant**: Nano (yolov8n) vs Small (yolov8s)
- **Precision**: FP32 (Full) vs FP16 (Half)

Benchmarking is performed using an automated suite with a 5-frame warmup and a 50-frame measurement phase. The infrastructure supports both live webcam input and synthetic frame fallbacks for headless environments.

## Results Summary

| Resolution | Model | Precision | FPS | Avg Latency (ms) | Observation |
|------------|-------|-----------|-----|------------------|-------------|
| 640x640    | YOLOv8n | FP32 | 8.66 | 115.54 | Standard precision |
| 416x416    | YOLOv8n | FP32 | 19.10 | 52.35 | Faster Inference |
| 640x640    | YOLOv8s | FP32 | 3.62 | 276.61 | Improved accuracy, higher latency |
| 640x640    | YOLOv8n | FP16 | 8.79 | 113.82 | Optimized for GPU, slower on CPU |

## Key Observations
- **Resolution Impact**: Reducing input resolution from 640 to 416 significantly improves throughput, doubling the FPS for the Nano model.
- **Model Scalability**: The Small variant (yolov8s) introduces over 2x the latency compared to Nano, emphasizing the compute-intensive nature of larger backbones on edge hardware.
- **Precision Characteristics**: FP16 inference on standard CPU hardware provides negligible performance gains (and may be slightly slower due to lack of optimized kernels), confirming its target use case for hardware-accelerated environments (NPU/GPU).

## Visualizations
Comparative visualizations are generated in `results/plots/`:
- `latency_vs_resolution.png`
- `fps_vs_resolution.png`
- `fps_vs_model.png`
- `latency_precision_comp.png`

## Reproducibility
To run the full benchmark suite:
1. Install dependencies: `pip install -r requirements.txt`
2. Execute benchmarks: `python3 experiments/run_all.py`

*Note: Use `export FORCE_SYNTHETIC=true` to run without a physical webcam.*

## Future Work
- INT8 quantization for ultra-low latency.
- Benchmarking on dedicated edge NPU hardware (e.g., Coral, Jetson).
- Integration of object tracking to evaluate system-wide latency.
