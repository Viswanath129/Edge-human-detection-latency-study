## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices. It provides a standardized benchmarking suite to evaluate YOLO-based models across different resolutions, model sizes, and precision levels.

## Research Question
How do input resolution, model architecture, and numerical precision affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline with a centralized benchmarking utility. The suite measures inference-only latency and FPS, supporting both live webcam input and headless synthetic frame generation for consistency.

## Experiments
- **Resolution comparison**: 640×640 vs 416×416.
- **Model size comparison**: YOLOv8 Nano vs YOLOv8 Small.
- **Precision comparison**: FP32 vs FP16 (requires CUDA-enabled hardware).

## Benchmarking Suite
To run the full suite of experiments:
```bash
python experiments/run_all.py
```
For headless environments:
```bash
export FORCE_SYNTHETIC=true
python experiments/run_all.py
```

## Preliminary Results
| Resolution | Model | Precision | FPS | Avg Latency (ms) | Notes |
|------------|-------|-----------|-----|------------------|-------|
| 640x640    | yolov8n | FP32      | 7.6 | 131.37           | Higher detection quality |
| 416x416    | yolov8n | FP32      | 19.76 | 50.51           | Faster Inference |
| 640x640    | yolov8s | FP32      | 4.37 | 228.44           | Balanced performance |

## Observations
- **Resolution**: Reducing input resolution from 640 to 416 improves FPS by ~2.6x on standard CPU hardware.
- **Model Size**: The Nano variant is ~1.7x faster than the Small variant at the same resolution.
- **Precision**: FP16 provides significant acceleration on supported NPUs/GPUs, though benefits are negligible on standard CPUs.

## Future Work
- INT8 quantization for edge deployment.
- Benchmarking on specific edge hardware (Jetson, Coral, etc.).
- mAP evaluation across different configurations.
