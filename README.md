# Edge Human Detection Benchmarking

This repository contains a benchmarking suite for studying latency-accuracy trade-offs of YOLO-based human detection on edge devices.

## Project Structure
- `experiments/`: Core benchmarking scripts and utilities.
  - `utils.py`: Centralized benchmarking and reporting logic.
  - `run_all.py`: Orchestrator to run the entire experiment suite.
  - `resolution_test.py`, `model_size_test.py`, `precision_test.py`: Specific experiment implementations.
  - `plot_results.py`: Visualization generation.
- `results/`: Output artifacts.
  - `tables/summary.csv`: Consolidated benchmark metrics.
  - `plots/`: Generated performance visualizations.
- `report/research_note.md`: Detailed analysis of experimental findings.

## Getting Started

### Prerequisites
- Python 3.8+
- Requirements: `torch`, `ultralytics`, `opencv-python`, `numpy`, `pandas`, `matplotlib`

### Installation
```bash
pip install -r requirements.txt
```

### Running Benchmarks
To execute the full suite of experiments and generate plots:
```bash
python3 experiments/run_all.py
```

Results will be saved to `results/tables/summary.csv` and visualizations in `results/plots/`.

## Results
Preliminary findings indicate that resolution scaling (640 to 416) is one of the most effective ways to achieve real-time performance on constrained hardware, providing over 2x speedup with YOLOv8n.

For a detailed breakdown, see [report/research_note.md](report/research_note.md).
