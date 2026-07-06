## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution and model configuration affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency. Our experiments cover variations in resolution, model architecture, and precision.

## Experimental Results

### Summary Table
| Resolution | Model | Precision | Average FPS | Avg Latency (ms) | Notes |
|------------|-------|-----------|-------------|------------------|-------|
| 640x640    | yolov8n | FP32 | 9.1 | 110.1 | Base configuration |
| 416x416    | yolov8n | FP32 | 19.9 | 50.3 | High throughput |
| 640x640    | yolov8s | FP32 | 4.3 | 234.8 | Higher accuracy, low FPS |

## Key Observations
- **Resolution**: Lowering resolution to 416x416 doubles FPS while maintaining reasonable human detection accuracy.
- **Model Size**: YOLOv8n is significantly faster than YOLOv8s (~2.1x), making it the preferred choice for real-time edge use.
- **Hardware**: CPU-based inference remains the bottleneck; NPU/GPU acceleration is required for high-resolution real-time performance.

## Usage
To reproduce the results:
```bash
# Install dependencies
pip install -r requirements.txt

# Run all benchmarks
python3 experiments/run_all.py
```

## Future Work
- Edge NPU benchmarking (TensorRT / OpenVINO)
- INT8 Quantization analysis
- Real-world accuracy validation on custom edge datasets
