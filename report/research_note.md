# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Experimental Setup
- **Hardware:** CPU-based inference (Simulated Edge Environment)
- **Model:** YOLOv8 (Nano and Small variants)
- **Task:** Real-time human detection
- **Metrics:** Average Latency (ms), Frames Per Second (FPS)

## Key Findings

### 1. Impact of Input Resolution
Reducing input resolution from 640x640 to 416x416 results in a significant performance boost.
- **Latency reduction:** ~52% improvement (92.30ms -> 43.85ms)
- **Throughput increase:** ~111% increase in FPS (10.40 -> 21.95)

While lower resolution improves speed, it typically results in reduced detection range and sensitivity for small objects.

### 2. Model Architecture Complexity
The transition from YOLOv8n (Nano) to YOLOv8s (Small) introduces substantial computational overhead.
- **Latency increase:** ~140% (92.30ms -> 221.51ms)
- **FPS drop:** ~57% (10.40 -> 4.44)

For edge devices without dedicated accelerators, the Nano variant is the only viable option for "real-time" (>10 FPS) performance.

### 3. Precision (FP32 vs FP16)
Benchmarking on standard CPU hardware showed no improvement for FP16, as modern CPUs often lack native half-precision acceleration equivalent to dedicated NPUs or GPUs. In our tests, the system automatically fell back to FP32 to maintain stability.

## Conclusion
For real-time human detection on edge CPUs, a configuration using **YOLOv8n at 416x416 resolution** provides the best balance, achieving ~22 FPS. If high precision is required for distant subjects, 640x640 is necessary but requires either acceptance of lower frame rates (~10 FPS) or specialized hardware acceleration.
