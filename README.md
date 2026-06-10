## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution and model configuration affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark performance using synthetic frames to ensure statistical reliability across environments.

## Experiments
- **Resolution Comparison**: 640×640 vs 416×416
- **Model Size Comparison**: YOLOv8n (Nano) vs YOLOv8s (Small)
- **Precision Analysis**: FP32 vs FP16

## Comprehensive Benchmark Results
| Resolution | Model | Precision | FPS | Avg Latency (ms) | Notes |
|------------|-------|-----------|-----|------------------|-------|
| 640x640 | YOLOv8n | FP32 | 9.25 | 108.12 | Higher accuracy baseline |
| 416x416 | YOLOv8n | FP32 | 18.32 | 54.6 | Optimized for speed |
| 640x640 | YOLOv8s | FP32 | 4.61 | 216.84 | Higher complexity |
| 640x640 | YOLOv8n | FP16 | 0.13 | 7483.74 | Hardware acceleration req. |

## Visualizations
Generated plots are available in `results/plots/`:
- `latency_vs_resolution.png`
- `fps_vs_resolution.png`
- `latency_precision_comp.png`
- `fps_vs_model.png`

## Observations
- Lower input resolution significantly improves inference latency without catastrophic accuracy loss for large objects (humans).
- YOLOv8n is the preferred choice for edge CPU deployment.
- FP16 inference requires dedicated hardware (NPU/GPU); CPU-based FP16 is non-viable.

## Limitations
- Single-device benchmarking
- Synthetic frame testing (no I/O overhead)

## Future Work
- Edge NPU benchmarking (Coral, Jetson)
- INT8 quantization analysis
- Real-world dataset validation
