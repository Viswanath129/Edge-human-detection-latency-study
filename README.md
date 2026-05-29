## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution, model configuration, and precision affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency. The suite supports synthetic frame generation for headless environments.

## Core Experiments
- **Resolution comparison**: 640×640 vs 416×416
- **Model size comparison**: YOLOv8n (Nano) vs YOLOv8s (Small)
- **Precision analysis**: FP32 vs FP16

## Key Results
| Configuration | Resolution | FPS | Avg Latency (ms) |
|---------------|------------|-----|------------------|
| YOLOv8n (FP32)| 640x640    | 9.45| 105.85           |
| YOLOv8n (FP32)| 416x416    | 20.83| 48.01           |
| YOLOv8s (FP32)| 640x640    | 4.26| 234.86           |

## Observations
- **Resolution**: Lowering resolution to 416x416 doubles the FPS on CPU.
- **Model Size**: YOLOv8n is significantly more viable for real-time edge use cases than YOLOv8s.
- **Hardware**: FP16 benchmarking shows that CPU-only environments should stick to FP32.

## Visualizations
Results plots are available in `results/plots/`, including:
- `fps_vs_resolution.png`
- `latency_vs_model.png`
- `fps_vs_precision.png`

## Reproducing Results
1. Install dependencies: `pip install -r requirements.txt`
2. Run all tests: `python3 experiments/run_all.py`
3. Generate plots: `cd results/plots && python3 plot_results.py`
