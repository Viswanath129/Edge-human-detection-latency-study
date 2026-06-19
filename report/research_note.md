# Research Note: Latency-Accuracy Trade-offs in YOLOv8 for Edge Devices

## Methodology
In this study, we analyzed the performance of YOLOv8-based human detection models on a standard compute environment (CPU-based benchmarking). We used a standardized benchmarking utility that includes:
- **Warmup Phase:** 5 frames to stabilize the inference pipeline.
- **Inference Timing:** Precise measurement of the inference step using `time.perf_counter()`.
- **Synthetic Fallback:** Support for headless benchmarking using synthetic frames when a webcam is unavailable.

## Experimental Results

### 1. Input Resolution Comparison (YOLOv8n)
| Resolution | Precision | Avg Latency (ms) | Average FPS |
|------------|-----------|------------------|-------------|
| 640x640    | FP32      | ~124.16          | ~8.05       |
| 416x416    | FP32      | ~51.41           | ~19.45      |

**Analysis:** Reducing the input resolution from 640 to 416 resulted in a ~58% reduction in inference latency, nearly doubling the frame rate. This is the most effective optimization for real-time performance on constrained devices, though it typically involves a trade-off in detecting small objects or distant subjects.

### 2. Model Variant Comparison (640x640, FP32)
| Model Variant | Avg Latency (ms) | Average FPS |
|---------------|------------------|-------------|
| YOLOv8n (Nano)| ~124.16          | ~8.05       |
| YOLOv8s (Small)| ~261.81          | ~3.82       |

**Analysis:** The YOLOv8s model is significantly heavier than YOLOv8n, with more than double the latency. While YOLOv8s offers higher accuracy, its use in real-time edge applications is limited unless hardware acceleration (e.g., NPU, GPU) is available.

### 3. Precision Level Comparison
*Note: FP16 results were not obtained in this CPU-only environment as FP16 is primarily designed for hardware-accelerated (CUDA) inference.*

## Conclusions
- **Resolution is the primary lever:** For real-time human detection on the edge, 416x416 resolution offers a superior balance of speed and performance.
- **Model Choice:** YOLOv8n is the clear choice for CPU-bound edge devices.
- **Hardware Acceleration:** Future work should explore INT8 quantization or FP16 on compatible hardware to further reduce latency.
