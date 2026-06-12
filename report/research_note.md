# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Executive Summary
This research investigates the performance characteristics of YOLOv8 variants for real-time human detection. Our benchmarks demonstrate that input resolution and model size are the primary levers for optimizing performance on resource-constrained devices.

## Performance Analysis
The benchmarking suite (located in `/experiments`) automates the collection of performance metrics across three dimensions:

1.  **Input Resolution**: 416x416 resolution offers a significant throughput advantage over 640x640, making it the preferred choice for high-speed edge applications where detection range requirements are moderate.
2.  **Model Complexity**: YOLOv8 Nano (yolov8n) consistently achieves real-time or near-real-time performance, whereas YOLOv8 Small (yolov8s) requires more robust hardware or lower resolutions to maintain acceptable frame rates.
3.  **Numerical Precision**: While FP16 is essential for leveraging hardware acceleration (CUDA/NPU), standard CPU environments show no gain and often a regression when forced to use half-precision.

## Architecture
The benchmarking infrastructure is designed for portability:
- `utils.py`: Centralized inference logic and statistical collection.
- `run_all.py`: Orchestrator for automated experimentation.
- `plot_results.py`: Visual analysis generation.

## Conclusion
For most edge human detection tasks, `yolov8n` at `416x416` resolution provides the optimal balance of responsiveness and accuracy.
