# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Executive Summary
This research explores the impact of various configuration parameters (resolution, model size, and precision) on the performance of YOLOv8-based human detection models. Our benchmarks indicate that input resolution and model depth are the primary drivers of inference latency on CPU-bound edge devices.

## Performance Metrics

### 1. Resolution Impact (Model: YOLOv8n, Precision: FP32)
| Resolution | FPS | Latency (ms) | Observation |
|------------|-----|--------------|-------------|
| 640x640    | 9.45 | 105.85       | Baseline high-resolution |
| 416x416    | 20.83| 48.01        | ~2.2x speedup with moderate accuracy loss |

### 2. Model Size Comparison (Resolution: 640x640, Precision: FP32)
| Model   | FPS | Latency (ms) | Observation |
|---------|-----|--------------|-------------|
| YOLOv8n | 9.45 | 105.85       | Highly optimized for edge |
| YOLOv8s | 4.26 | 234.86       | ~2.2x slower, better for complex scenes |

### 3. Precision Effects (Resolution: 640x640, Model: YOLOv8n)
| Precision | FPS | Latency (ms) | Observation |
|-----------|-----|--------------|-------------|
| FP32      | 9.45 | 105.85       | Standard CPU performance |
| FP16      | 0.14 | 7260.02      | Non-representative on CPU (Requires GPU/NPU) |

## Technical Observations
- **Resolution Scaling:** Reducing resolution from 640 to 416 provides a near-linear speedup in inference time, making it the most effective optimization for real-time requirements on limited hardware.
- **Model Depth:** Moving from Nano to Small variants significantly increases the computational burden. For human detection, the Nano model often provides sufficient accuracy at a much higher frame rate.
- **Precision:** FP16 inference is not recommended for standard CPU execution as it relies on specific hardware acceleration (CUDA/NPU). On CPU, it leads to significant performance degradation.

## Conclusion
For real-time human detection on edge devices, a combination of **YOLOv8n** at **416x416** resolution provides the best balance of responsiveness and detection capability.
