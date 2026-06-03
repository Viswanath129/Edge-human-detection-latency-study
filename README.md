## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution, model size, and inference precision affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency. Our benchmarking utility includes:
- Warmup phase (5 frames)
- Synthetic frame fallback for headless environments
- `time.perf_counter()` for high-precision timing

## Experiments
- **Resolution comparison**: 640×640 vs 416×416
- **Model size comparison**: YOLOv8 Nano vs YOLOv8 Small
- **Precision comparison**: FP32 vs FP16

## Results Summary
| Resolution | Model | Precision | FPS | Avg Latency (ms) | Notes |
|------------|-------|-----------|-----|------------------|-------|
| 640x640    | yolov8n | FP32 | 9.67 | 103.39 | Standard baseline |
| 416x416    | yolov8n | FP32 | 21.06 | 47.49 | High-speed optimized |
| 640x640    | yolov8s | FP32 | 4.28 | 233.86 | Higher capacity, slower inference |
| 640x640    | yolov8n | FP16 | 0.14 | 7344.49 | Non-optimal on CPU |

## Key Observations
- **Resolution**: Reducing resolution from 640 to 416 more than doubles the FPS (9.67 to 21.06).
- **Model Size**: Moving from Nano to Small results in a ~54% latency increase.
- **Precision**: FP16 inference on standard CPUs is significantly slower than FP32 due to lack of hardware acceleration for half-precision arithmetic. This confirms that FP16 should only be used on supported GPUs/NPUs.

## Limitations
- **Hardware**: Benchmarks conducted on standard CPU; results will vary significantly on Edge NPUs (e.g., Jetson, Coral).
- **Synthetic Data**: Headless validation uses synthetic frames; real-world I/O overhead from webcams might slightly reduce FPS.

## Future Work
- Edge NPU benchmarking (Jetson Orin/Xavier)
- INT8 Quantization tests
- Accuracy benchmarking on COCO/Human datasets
