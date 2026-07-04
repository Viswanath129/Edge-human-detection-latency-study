# Edge Human Detection: Latency–Accuracy Analysis

This repository contains a research project studying the latency–accuracy trade-offs for real-time human detection models on edge devices.

## Research Question
How do input resolution, model architecture, and precision affect inference latency and throughput under compute constraints?

## Project Structure
- `experiments/`: Benchmarking scripts for resolution, model size, and precision.
- `results/`: Standardized CSV tables and comparative visualization plots.
- `report/`: Detailed research note and performance analysis.

## Key Performance Results

| Model | Resolution | Precision | FPS | Latency (ms) |
|-------|------------|-----------|-----|--------------|
| YOLOv8n | 640x640 | FP32 | 9.9 | 100.6 |
| YOLOv8n | 416x416 | FP32 | 21.3 | 47.0 |
| YOLOv8s | 640x640 | FP32 | 4.6 | 217.3 |

## Visualizations
Generated plots comparing performance across different configurations can be found in `results/plots/`.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run full suite: `python experiments/run_all.py`
   - Use `export FORCE_SYNTHETIC=true` if no webcam is available.

## Observations
- **Resolution is the most effective lever:** Switching to 416x416 more than doubled the throughput.
- **Model Choice:** YOLOv8n is strictly required for real-time targets on CPU.
- **Hardware Acceleration:** FP16 requires specific hardware (CUDA/NPU) for actual speedup.

## Future Work
- Edge NPU benchmarking (Coral, OpenVINO)
- INT8 Quantization evaluation
- Mean Average Precision (mAP) vs. Latency Pareto curves
