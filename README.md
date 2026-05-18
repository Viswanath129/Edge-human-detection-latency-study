## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution, model configuration, and precision affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency. Our benchmarking suite supports both live webcam inference and headless execution using synthetic frames for automated testing.

## Experiments
- **Resolution comparison**: 640×640 vs 416×416 (`experiments/resolution_test.py`)
- **Model variant comparison**: YOLOv8n (Nano) vs YOLOv8s (Small) (`experiments/model_size_test.py`)
- **Precision comparison**: FP32 vs FP16 (`experiments/precision_test.py`)

## Preliminary Results
| Resolution | FPS | Avg Latency (ms) | Notes |
|------------|-----|------------------|-------|
| 640        | 7.6  | 110.0               | Higher accuracy |
| 416        | 14.2  | 65.0               | Faster inference |

## Benchmarking Suite Usage
To run the benchmarks in a headless environment:
```bash
export HEADLESS=true
python experiments/resolution_test.py
python experiments/model_size_test.py
python experiments/precision_test.py
```

## Observations
- Lower input resolution significantly improves inference latency.
- Accuracy degradation is moderate for human detection tasks.
- Model variant selection (e.g., Nano vs Small) provides a primary lever for balancing throughput and precision.

## Limitations
- Limited dataset size for accuracy validation.
- FP16 performance is hardware-dependent (optimized for CUDA/NPU).
- Single-device benchmarking.

## Future Work
- Edge NPU-specific benchmarking.
- TensorRT and OpenVINO optimization paths.
- Quantization (INT8) impact analysis.
