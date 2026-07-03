# Engineering Note: Latency-Accuracy Trade-offs in Edge YOLO Inference

## Experimental Setup
- **Hardware**: Edge-representative CPU/GPU environment.
- **Model**: Ultralytics YOLOv8 (Nano and Small variants).
- **Framework**: PyTorch with standard FP32 and optimized FP16 (CUDA) precision.
- **Metric**: Inference-only latency and FPS (excluding I/O overhead).

## Key Dimensions Analyzed

### 1. Input Resolution Impact
Reducing input resolution from 640x640 to 416x416 significantly decreases computational complexity.
- **Observation**: Lowering resolution yields a non-linear improvement in FPS, making it the most effective lever for real-time performance on restricted hardware.
- **Trade-off**: Small object detection accuracy typically degrades at lower resolutions.

### 2. Model Architecture (Nano vs. Small)
Comparing `yolov8n` (Nano) and `yolov8s` (Small) highlights the cost of model depth and width.
- **Observation**: YOLOv8n offers the best performance-to-latency ratio for edge devices, while YOLOv8s provides better feature extraction at the cost of increased per-frame latency.

### 3. Precision Optimization (FP32 vs. FP16)
Half-precision (FP16) inference leverages specialized hardware (tensor cores) to accelerate computation.
- **Observation**: In CUDA-enabled environments, FP16 reduces latency and memory footprint with negligible impact on detection accuracy.

## Performance Summary
Detailed metrics are consolidated in `results/tables/summary.csv` and visualized in `results/plots/`.
