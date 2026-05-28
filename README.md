## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices. We analyze how input resolution, model architecture, and numerical precision affect performance.

## Research Question
How do input resolution, model configuration, and inference precision affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency. Benchmarks are performed using `ultralytics` YOLOv8 implementations. For headless environments, the pipeline supports synthetic frame generation to simulate live video inference.

### Key Metrics
- **Average Latency (ms)**: Time taken for a single forward pass.
- **FPS**: Frames processed per second, accounting for end-to-end overhead.

## Experiments & Results

### 1. Resolution Comparison
Comparing YOLOv8n at different input dimensions.

| Resolution | Model | Precision | FPS | Avg Latency (ms) |
|------------|-------|-----------|-----|------------------|
| 640x640    | Nano  | FP32      | 8.3 | 116.3            |
| 416x416    | Nano  | FP32      | 18.0| 53.8             |

### 2. Precision Comparison
Impact of half-precision (FP16) on inference.
*Note: FP16 results on standard CPU are significantly slower due to lack of specialized hardware acceleration.*

| Resolution | Model | Precision | FPS | Avg Latency (ms) |
|------------|-------|-----------|-----|------------------|
| 640x640    | Nano  | FP32      | 8.1 | 119.7            |
| 640x640    | Nano  | FP16      | 0.1 | 7250.8           |

### 3. Model Size Comparison
Performance trade-offs between different YOLOv8 variants.

| Resolution | Model | Precision | FPS | Avg Latency (ms) |
|------------|-------|-----------|-----|------------------|
| 640x640    | Nano  | FP32      | 8.2 | 118.5            |
| 640x640    | Small | FP32      | 3.8 | 260.3            |

## Observations
- **Resolution**: Reducing input resolution from 640 to 416 nearly doubles the FPS.
- **Model Size**: Moving from 'Nano' to 'Small' increases latency by ~2.2x, suggesting a significant cost for the improved accuracy.
- **Precision**: FP16 inference requires hardware support (e.g., CUDA/NPU) to be effective; on CPU, it is not recommended.

## Future Work
- Edge NPU benchmarking (e.g., Hailo-8, Coral)
- INT8 quantization and hardware-specific optimization (TensorRT, OpenVINO)
- Formal mAP evaluation on human detection datasets
