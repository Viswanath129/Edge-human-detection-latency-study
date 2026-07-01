# Edge YOLO Human Detection: Latency-Accuracy Analysis

This repository explores the trade-offs between inference latency and detection accuracy for YOLO-based human detection models on edge devices.

## Research Questions
1. How does input resolution (640 vs 416) impact real-time throughput?
2. What is the performance penalty of moving from Nano to Small model architectures?
3. Does half-precision (FP16) inference provide benefits on edge CPU/GPU hardware?

## Methodology
Inference is benchmarked using the `ultralytics` YOLOv8 implementation. The suite includes:
- **Warmup:** 5-frame initialization to stabilize hardware clocks.
- **Metrics:** Inference-only latency (ms) and Frames Per Second (FPS).
- **Environment:** Support for physical webcam and headless synthetic frame fallback.

## Results Summary

| Model | Resolution | Precision | FPS | Avg Latency (ms) |
|-------|------------|-----------|-----|------------------|
| YOLOv8n | 640x640 | FP32 | ~10 | ~100 |
| YOLOv8n | 416x416 | FP32 | ~22 | ~45 |
| YOLOv8s | 640x640 | FP32 | ~5 | ~210 |

## Visualizations
Comparative plots are generated in the `results/plots/` directory:
- `latency_vs_resolution.png`
- `fps_vs_resolution.png`
- `latency_vs_model.png`

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run all experiments: `python3 experiments/run_all.py`
3. View analysis: See `report/research_note.md` for detailed findings.
