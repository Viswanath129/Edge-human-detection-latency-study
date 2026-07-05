# YOLO Edge Performance Analysis

This repository contains a research suite for studying latency-accuracy trade-offs for YOLO-based human detection on edge devices.

## Research Question
How do input resolution, model architecture, and precision affect inference latency and throughput under compute constraints?

## Experimental Results

| Resolution | Model | Precision | FPS | Avg Latency (ms) | Notes |
|------------|-------|-----------|-----|------------------|-------|
| 640x640    | Nano  | FP32      | 8.40| 119.10           | Standard baseline |
| 416x416    | Nano  | FP32      | 19.56| 51.12            | Optimized for speed |
| 640x640    | Small | FP32      | 3.94| 253.53           | Higher capacity |

## Key Observations
- **Resolution**: Lowering resolution to 416x416 nearly doubles the FPS on CPU.
- **Model Size**: The Small variant is ~2x slower than the Nano variant at the same resolution.
- **Hardware**: For real-time performance on CPU, Nano at 416px is the recommended configuration.

## Project Structure
- `experiments/`: Benchmarking scripts (resolution, model size, precision).
- `results/`: CSV tables and comparative plots.
- `report/`: Detailed research notes and analysis.

## Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Benchmarks
To run the full suite and generate plots:
```bash
python3 experiments/run_all.py
```
*Note: Set `FORCE_SYNTHETIC=true` for headless environments.*

## Future Work
- INT8 Quantization benchmarks.
- Evaluation on dedicated Edge NPUs (Coral, Jetson).
- Object-specific accuracy vs. resolution trade-off curves.
