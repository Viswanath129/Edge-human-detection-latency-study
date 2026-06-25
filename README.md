## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution and model configuration affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark performance using a standardized suite of experiments.

## Usage
To run the full benchmark suite:
```bash
export FORCE_SYNTHETIC=true # For headless/no-webcam environments
python3 experiments/run_all.py
```

## Experiments
- **Resolution comparison**: 640×640 vs 416×416
- **Model scale comparison**: YOLOv8n (Nano) vs YOLOv8s (Small)
- **Precision comparison**: FP32 vs FP16

## Results Summary
Detailed findings can be found in [report/research_note.md](report/research_note.md).

| Resolution | Model | Precision | FPS | Avg Latency (ms) | Notes |
|------------|-------|-----------|-----|------------------|-------|
| 640x640    | yolov8n | FP32      | 8.3 | 116.6            | Base Nano |
| 416x416    | yolov8n | FP32      | 18.6| 52.1             | High Speed |
| 640x640    | yolov8s | FP32      | 3.8 | 258.7            | Better Accuracy |

## Observations
- Lower input resolution significantly improves inference latency (~55% reduction).
- Model scaling from Nano to Small more than doubles the latency on CPU.
- 416x416 resolution is recommended for real-time edge applications.

## Future Work
- INT8 Quantization for further acceleration.
- Benchmarking on dedicated edge NPUs (Coral, Jetson).
- Formal accuracy (mAP) evaluation on standard datasets.
