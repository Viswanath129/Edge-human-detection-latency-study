# Edge Inference Performance Analysis

## Executive Summary
This analysis explores the trade-offs between inference latency and detection configuration for YOLOv8 models on edge-simulated environments. We evaluated three primary dimensions: input resolution, model complexity (scale), and numerical precision.

## Experimental Results

### 1. Resolution Impact
Reducing input resolution from 640x640 to 416x416 yielded the most significant performance gain.
- **640x640:** ~115ms latency (~8.7 FPS)
- **416x416:** ~54ms latency (~18.6 FPS)
- **Impact:** ~53% reduction in latency. While lower resolution reduces the detection range for small objects, it is essential for achieving near real-time performance on constrained hardware.

### 2. Model Scale Comparison
Comparison between YOLOv8n (Nano) and YOLOv8s (Small) at 640x640 resolution.
- **YOLOv8n:** ~115ms latency
- **YOLOv8s:** ~281ms latency
- **Impact:** The Small model is ~2.4x slower than the Nano model. For most edge human detection tasks, the Nano model provides an optimal balance, as the accuracy gains from the Small variant often do not justify the severe throughput penalty.

### 3. Precision (FP32 vs FP16)
Benchmarking FP16 on CPU-bound environments showed marginal differences, as expected.
- **FP32:** ~115ms
- **FP16 (CPU fallback):** ~101ms
- **Observation:** Significant acceleration from FP16 requires hardware-level support (CUDA cores or specialized NPUs). On standard CPU edge devices, optimization efforts should focus on resolution and quantization (INT8) rather than half-precision alone.

## Conclusions and Recommendations
1. **Prioritize Resolution Scaling:** Adjusting `imgsz` to 416 or 320 is the most effective way to meet real-time requirements (>15 FPS).
2. **Standardize on Nano variants:** YOLOv8n should be the baseline for edge deployment.
3. **Hardware Acceleration:** Future work should target INT8 quantization via OpenVINO or TensorRT to further bridge the gap between CPU and NPU performance.
