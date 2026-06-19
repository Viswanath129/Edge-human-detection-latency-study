# YOLOv8 Latency–Accuracy Trade-off Analysis

This repository contains a research suite designed to analyze the latency and accuracy trade-offs of YOLOv8 models for real-time human detection on edge devices.

## Project Structure
- `experiments/`: Benchmarking scripts and utilities.
  - `utils.py`: Centralized benchmarking and reporting logic.
  - `run_all.py`: Orchestrator to execute all experiments.
  - `resolution_test.py`: Compares 640x640 vs 416x416.
  - `model_size_test.py`: Compares YOLOv8n (Nano) vs YOLOv8s (Small).
  - `precision_test.py`: Compares FP32 vs FP16 precision.
- `results/`: Standardized output data and visualizations.
  - `tables/summary.csv`: Consolidated results from all runs.
  - `plots/`: Generated performance charts.
- `report/`: Detailed findings and research notes.

## Installation
```bash
pip install -r requirements.txt
```

## Running Experiments
To run the entire suite and generate plots:
```bash
python3 experiments/run_all.py
```

For headless environments or to skip webcam detection:
```bash
export FORCE_SYNTHETIC=true
python3 experiments/run_all.py
```

## Key Findings
- **Resolution Impact:** Reducing input size to 416x416 provides a ~2x speedup compared to 640x640.
- **Model Efficiency:** YOLOv8n is highly optimized for edge deployment, outperforming YOLOv8s by over 2x in inference speed on CPU.
- **Optimization Path:** For real-time performance on edge devices, combining lower resolution (416px) with the Nano model variant is recommended.

For a deep dive into the data, see [report/research_note.md](report/research_note.md).
