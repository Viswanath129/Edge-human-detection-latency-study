# Research Note: YOLOv8 Edge Benchmarking for Human Detection

## Overview
This document summarizes the findings from the latency-accuracy trade-off analysis performed on various YOLOv8 configurations, targeting edge deployment for human detection. The experiments study performance differences across input resolution and model size to guide edge hardware selection and optimization.

## Benchmark Results

| Resolution | Model | Precision | FPS | Avg Latency (ms) | Observation |
|------------|-------|-----------|-----|------------------|-------------|
| 640x640 | YOLOv8n | FP32 | 9.73 | 102.79 | Lightweight edge model |
| 416x416 | YOLOv8n | FP32 | 16.10 | 62.11 | Faster Inference |
| 640x640 | YOLOv8s | FP32 | 4.21 | 237.41 | Balanced accuracy/speed |

## Key Observations

1. **Resolution Impact**: Decreasing the input size from 640x640 to 416x416 increases throughput by ~65% (from 9.73 FPS to 16.10 FPS) and decreases latency by ~39.6% (from 102.79 ms to 62.11 ms). This highlights input downscaling as one of the most effective knobs to trade off precision for latency budget on resource-constrained hardware.
2. **Model Complexity**: Elevating model capacity from YOLOv8 Nano (`YOLOv8n`) to Small (`YOLOv8s`) causes latency to increase by over 2.3x (from 102.79 ms to 237.41 ms) and cuts FPS by ~56.7% (from 9.73 FPS to 4.21 FPS). While `YOLOv8s` is expected to have higher detection robustness, its computational demand makes it challenging for real-time edge processing without acceleration.
3. **Precision Testing (FP16 fallback)**: Standard FP16 checks were verified. When CUDA is unavailable on CPU environments, running in FP16 falls back to FP32 or runs sub-optimally. Hence, precision-aware inference experiments should prioritize hardware-accelerated environments (such as CUDA/NPU) to avoid non-representative performance.

## Conclusion
For real-time scenarios on standard edge CPUs, **YOLOv8n with 416x416 input resolution** provides the most viable operating point, yielding ~16 FPS and latency close to 60 ms. For stationary applications where high accuracy is crucial and lower frame rates are acceptable, **YOLOv8s with 640x640 input resolution** remains a viable candidate if paired with hardware-accelerated inference engines.
