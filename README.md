# YOLO Edge Benchmarking

This repository explores the latency–accuracy trade-offs for YOLO-based human detection on edge devices.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running Benchmarks
To run the full suite of experiments (resolution, model size, and precision):
```bash
python3 experiments/run_all.py
```
*Note: Use `export FORCE_SYNTHETIC=true` if running in a headless environment without a webcam.*

## Experiments
- **Resolution**: 640x640 vs 416x416.
- **Model Size**: Nano (yolov8n) vs Small (yolov8s).
- **Precision**: FP32 vs FP16 (requires CUDA).

## Results Summary
Current results are stored in `results/tables/summary.csv` and visualized in `results/plots/`.

| Resolution | Model | Precision | FPS | Latency (ms) |
|------------|-------|-----------|-----|--------------|
| 640x640    | YOLOv8n | FP32      | 9.8 | 102.0        |
| 416x416    | YOLOv8n | FP32      | 21.6 | 46.3         |
| 640x640    | YOLOv8s | FP32      | 4.3 | 231.7        |

For detailed analysis, see [Research Note](report/research_note.md).
