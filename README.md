## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution and model configuration affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency. Our suite supports:
- Multiple input resolutions (640, 416)
- Model variants (Nano, Small)
- Precision levels (FP32, FP16)
- Headless benchmarking via synthetic frame fallback

## Execution
To run the full benchmark suite:
```bash
export FORCE_SYNTHETIC=true
python3 experiments/run_all.py
```

## Observations
- Lower input resolution significantly improves inference latency.
- FP16 provides hardware-accelerated speedups on supported devices.
- YOLOv8n remains the preferred choice for strict edge constraints.

## Future Work
- Edge NPU-specific optimizations (TensorRT, OpenVINO)
- Power consumption analysis during inference
- Quantization (INT8) impact studies
