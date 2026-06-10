# Research Note: YOLOv8 Edge Benchmarking

## Overview
This document summarizes the findings from the latency-accuracy trade-off analysis performed on various YOLOv8 configurations.

## Benchmark Results

| Resolution | Model | Precision | FPS | Avg Latency (ms) | Observation |
|------------|-------|-----------|-----|------------------|-------------|
| 640x640 | YOLOv8n | FP32 | 9.25 | 108.12 | Standard precision |
| 416x416 | YOLOv8n | FP32 | 18.32 | 54.6 | Faster Inference |
| 640x640 | YOLOv8s | FP32 | 4.61 | 216.84 | Balanced accuracy/speed |
| 640x640 | YOLOv8n | FP16 | 0.13 | 7483.74 | Non-representative (CPU) |

## Key Observations

1. **Resolution Impact**: Reducing input resolution from 640 to 416 nearly doubles the FPS (~9.3 to ~18.3), demonstrating a linear-like speedup suitable for less demanding detection tasks.
2. **Model Complexity**: Moving from Nano to Small model variants results in a ~50% reduction in frame rate. The latency increases from ~108ms to ~217ms, which may be acceptable for applications requiring higher precision.
3. **Precision Pitfalls**: Half-precision (FP16) inference is significantly slower on the current CPU-only environment. This highlights the necessity of hardware acceleration (e.g., CUDA/NPU) for FP16/INT8 optimizations to be effective.

## Conclusion
For real-time applications on edge CPUs, YOLOv8n with 416x416 resolution provides the best balance, achieving nearly 20 FPS. YOLOv8s is better suited for stationary monitoring where high accuracy is prioritized over high frame rates.
