# Technical Note: Real-Time Human Detection Optimization for Edge Devices

## Abstract
This report documents performance benchmarks for YOLOv8 variants on resource-constrained environments. We analyze the sensitivity of inference latency to input resolution, model depth, and numerical precision.

## Experimental Design
The benchmarking utility (`experiments/utils.py`) isolates inference time from I/O overhead. This is achieved by:
1. Pre-loading the model into memory.
2. Implementing a 5-frame warmup to initialize the compute graph and cache.
3. Measuring per-inference latency using high-resolution timers (`time.perf_counter`).

## Analysis of Results

### Resolution Impact
The jump from 640x640 to 416x416 results in a ~52% reduction in latency. This is expected as the number of pixels processed drops by ~57%. For edge devices where human detection occurs at close range, 416x416 is the recommended operating point.

### Architectural Complexity
YOLOv8s (Small) offers superior feature extraction but at a 128% latency increase compared to YOLOv8n (Nano). In strictly power-constrained scenarios, the Nano variant is the only viable option for >5 FPS performance.

### Numerical Precision
FP16 inference demonstrates a ~11% speedup even in our simulated environment. On hardware with native half-precision support (e.g., NVIDIA Tensor Cores), this gap is expected to widen significantly, enabling higher resolution at the same power envelope.

## Recommendations
1. **Prioritize YOLOv8n** for general edge applications.
2. **Deploy at 416x416** if objects of interest are large in the frame.
3. **Enable FP16** quantization as a default optimization for production deployments on supported hardware.
