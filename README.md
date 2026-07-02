# Edge Human Detection: Latency-Accuracy Analysis

Research project studying the latency–accuracy trade-offs of YOLO-based human detection models on edge devices.

## Overview
This repository contains a benchmarking suite for analyzing how different model configurations (resolution, architecture, precision) affect real-time performance.

## Experiments
We evaluate the following dimensions:
- **Resolution**: 640x640 vs 416x416
- **Model Size**: YOLOv8n (Nano) vs YOLOv8s (Small)
- **Precision**: FP32 vs FP16

## Methodology
Inference performance is measured using `time.perf_counter()` over a loop of 50 frames, with a 5-frame warmup phase. The system supports live webcam input or synthetic frame generation for headless environments.

## Results Summary
| Resolution | Model | Precision | Average FPS | Avg Latency (ms) |
|------------|-------|-----------|-------------|------------------|
| 640x640    | yolov8n.pt | FP32 | 9.30 | 103.95 |
| 416x416    | yolov8n.pt | FP32 | 19.71 | 48.24 |
| 640x640    | yolov8s.pt | FP32 | 4.18 | 235.81 |

*Note: Results obtained on CPU-based sandbox environment.*

## Project Structure
- `experiments/`: Benchmarking scripts and utilities.
- `results/`: CSV tables and generated performance plots.
- `report/`: Detailed research notes and analysis.

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the full benchmark suite:
   ```bash
   python3 experiments/run_all.py
   ```
   *To run in headless mode without a webcam:*
   ```bash
   export FORCE_SYNTHETIC=true && python3 experiments/run_all.py
   ```
