## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices, specifically analyzing YOLOv8 variants.

## Research Question
How do input resolution, model configuration, and inference precision affect latency and throughput under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency. The suite supports live webcam inference with an automatic fallback to synthetic frames for headless or non-camera environments.

### Benchmarking Process
1. **Warmup Phase**: 5 frames are processed to stabilize hardware/software states.
2. **Inference Phase**: 20 frames are processed using `time.perf_counter()` to measure pure inference latency.
3. **Metrics**: Both Average Latency (ms) and FPS are recorded.

## Multi-Dimensional Analysis
We evaluate three primary axes:
- **Resolution**: 640x640 vs 416x416
- **Model Size**: Nano (yolov8n) vs Small (yolov8s)
- **Precision**: FP32 vs FP16

## Latest Results
Comprehensive results are maintained in `results/tables/summary.csv`.

| Resolution | Model | Precision | FPS | Avg Latency (ms) | Observation |
|------------|-------|-----------|-----|------------------|-------------|
| 640x640    | yolov8n | FP32      | 9.26 | 107.97           | Standard precision |
| 416x416    | yolov8n | FP32      | 20.76 | 48.17            | Faster Inference |
| 640x640    | yolov8s | FP32      | 4.25 | 235.54           | Improved accuracy |
| 640x640    | yolov8n | FP16      | 0.14 | 7275.21          | Optimized for GPU/NPU |

*Note: FP16 performance on CPU is significantly slower than FP32 due to lack of native hardware acceleration.*

## Visualizations
Comparative plots are generated automatically in `results/plots/`:
- `latency_vs_resolution.png`
- `fps_vs_resolution.png`
- `fps_vs_model.png`
- `latency_precision_comp.png`

## Running Experiments
To execute the entire benchmark suite:
```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python3 experiments/run_all.py
```
To force synthetic frames in a headless environment, the script automatically sets `FORCE_SYNTHETIC=true`.

## Future Work
- Edge NPU benchmarking (e.g., Coral TPU, Intel OpenVINO)
- INT8 quantization analysis
- Tracking performance on specialized ARM architectures
