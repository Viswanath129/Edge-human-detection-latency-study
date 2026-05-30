## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution, model variant, and precision affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency. Benchmarks are performed using both live video and synthetic frame generation for headless environments.

## Comprehensive Results
| Resolution | Model | Precision | FPS | Avg Latency (ms) | Notes |
|------------|-------|-----------|-----|------------------|-------|
| 640x640    | Nano  | FP32      | 9.11| 109.79           | Base performance |
| 416x416    | Nano  | FP32      | 20.97| 47.68           | Recommended for Real-time |
| 640x640    | Nano  | FP16      | 0.14| 7121.94          | Slow on CPU (requires NPU) |
| 640x640    | Small | FP32      | 4.50| 222.21           | Higher accuracy, lower FPS |

## Key Observations
- **Resolution is the primary lever:** Dropping from 640px to 416px more than doubles the FPS.
- **CPU vs Accelerator:** FP16 is significantly slower on CPU; it should only be used with GPU/NPU acceleration.
- **Model Selection:** YOLOv8n (Nano) is the most suitable variant for real-time edge applications on limited hardware.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run all benchmarks: `python3 experiments/run_all.py` (Note: sets `FORCE_SYNTHETIC=true` by default)
3. Generate plots: `python3 results/plots/plot_results.py`

## Future Work
- Edge NPU-specific benchmarking (e.g., Coral, Hailo)
- INT8 Quantization tests
- Multi-object detection performance analysis
