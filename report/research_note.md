# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Experimental Setup
- **Hardware**: CPU-based benchmarking (Synthetic fallback for headless environment)
- **Models**: YOLOv8 Nano (yolov8n), YOLOv8 Small (yolov8s)
- **Resolutions**: 640x640, 416x416
- **Precision**: FP32 (Standard)

## Key Findings

### 1. Impact of Input Resolution
Reducing input resolution from 640 to 416 significantly improves throughput.
- **YOLOv8n (640x640)**: ~9 FPS, ~110ms latency
- **YOLOv8n (416x416)**: ~20 FPS, ~50ms latency
- **Takeaway**: A 35% reduction in resolution yields a >100% increase in FPS on CPU.

### 2. Model Architecture Complexity
Moving from the Nano to the Small variant of YOLOv8 results in a substantial performance penalty.
- **YOLOv8n (640x640)**: ~9 FPS
- **YOLOv8s (640x640)**: ~4.3 FPS
- **Takeaway**: YOLOv8s is approximately 2.1x slower than YOLOv8n, making it less suitable for real-time applications on constrained edge CPUs.

### 3. Precision Optimization
FP16 (Half-precision) inference was tested but requires CUDA-capable hardware for performance gains. On standard CPU environments, FP16 either falls back to FP32 or executes with significant overhead.

## Conclusion
For real-time human detection on edge devices without dedicated GPU acceleration, **YOLOv8n at 416x416 resolution** provides the most viable balance, maintaining usable detection frequency (>15 FPS) while keeping latency low.
