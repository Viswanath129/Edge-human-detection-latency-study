# Edge Human Detection: Latency–Accuracy Trade-off Analysis

This repository contains a research-focused study on the trade-offs between inference speed and detection accuracy for YOLOv8 models on edge devices.

## Project Overview
We evaluate how input resolution, model architecture size, and numerical precision affect the performance of real-time human detection systems.

## Key Features
- **Automated Benchmarking**: Scripts to measure latency and FPS across various configurations.
- **Hardware-Aware Fallbacks**: Supports both real webcam input and synthetic frame generation for headless environments.
- **Multi-Dimensional Analysis**:
  - **Resolution**: 640x640 vs 416x416.
  - **Model Size**: YOLOv8 Nano vs YOLOv8 Small.
  - **Precision**: FP32 vs FP16 (with hardware check).
- **Data Visualization**: Automated generation of performance comparison charts.

## Results Summary

| Configuration | Resolution | Precision | FPS | Latency (ms) |
|---------------|------------|-----------|-----|--------------|
| YOLOv8n       | 416x416    | FP32      | 20.1| 49.9         |
| YOLOv8n       | 640x640    | FP32      | 9.4 | 106.9        |
| YOLOv8s       | 640x640    | FP32      | 4.0 | 248.5        |

*Note: Benchmarks were performed using synthetic frame generation in a controlled environment.*

## Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Benchmarks
You can run individual tests or all of them:
```bash
# Set for headless/simulated environments
export FORCE_SYNTHETIC=true
export PYTHONPATH=$PYTHONPATH:$(pwd)/experiments

python3 experiments/resolution_test.py
python3 experiments/model_size_test.py
python3 experiments/precision_test.py
```

### 3. Generate Plots
```bash
python3 results/plots/plot_results.py
```

## Detailed Findings
For a deep dive into the experimental results and analysis, see the [Research Note](report/research_note.md).

## Future Work
- Integration with TensorRT for NVIDIA Jetson devices.
- Quantization (INT8) performance analysis.
- Multi-object detection benchmarks.
