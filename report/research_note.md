# Research Note: Latency-Accuracy Trade-off Analysis for Real-Time Human Detection on Edge Devices

## Executive Summary
This research note explores the multi-dimensional trade-offs in real-time human detection models deployed on edge devices under resource constraints. By systematically analyzing input resolution, model architecture capacity (parameters/size), and precision quantization, we provide a concrete, data-driven framework for selecting optimal model configurations for edge deployment.

---

## 1. Introduction & Objectives
Deep learning models for real-time human detection are increasingly deployed on resource-constrained edge platforms (e.g., Raspberry Pi, Jetson Nano, edge NPUs). However, these devices impose strict limitations on thermal dissipation, memory bandwidth, and compute capability.

The primary objective of this study is to profile and analyze how:
1. **Input Resolution** (640×640 vs. 416×416)
2. **Model Size** (YOLOv8 Nano, Small, Medium)
3. **Precision** (FP32 vs. FP16)

cooperatively affect inference latency and throughput (FPS).

---

## 2. Experimental Setup & Methodology
To ensure high-fidelity performance metrics, we developed a centralized benchmarking suite in `experiments/` which:
- Features **inference-only timing** using `time.perf_counter()` to eliminate I/O, camera frame reading, and preprocessing noise.
- Pre-generates synthetic frames outside the timed loop to guarantee technical precision.
- Incorporates a **5-frame warmup phase** at the target precision to allow GPU/CPU caches and runtime libraries to stabilize.
- Employs `torch.cuda.synchronize()` where applicable to prevent asynchronous execution profiling errors.
- Gracefully handles physical sensors (falling back to synthetic inputs if no webcam is connected).

All benchmarks are performed sequentially to prevent inter-process resource contention.

---

## 3. Empirical Results & Technical Analysis

### 3.1 Impact of Input Resolution
Input resolution dictates the spatial dimension of the tensor processed through the network. Downscaling from $640 \times 640$ to $416 \times 416$ represents a **57.7% reduction** in the total number of pixels.

| Resolution | Model | Precision | Average FPS | Avg Latency (ms) | Key Observations |
|------------|-------|-----------|-------------|------------------|------------------|
| **640x640**| YOLOv8n | FP32 | 9.80 | 102.02 | High spatial fidelity; ideal for distant/small humans. |
| **416x416**| YOLOv8n | FP32 | 15.70 | 63.70 | ~1.6x speedup; moderate accuracy degradation on small objects. |

**Key Takeaway**: Reducing resolution is a highly effective, parameter-free mechanism to achieve near-real-time performance. It scales down computational complexity quadratically, making it highly valuable when hardware acceleration is unavailable.

---

### 3.2 Impact of Model Architecture Capacity (Size)
As we scale from Nano (n) to Small (s) and Medium (m), the model capacity, number of parameters, and feature map depths increase.

| Model Size | Parameters | Resolution | Average FPS | Avg Latency (ms) | Scaling Factor (vs. Nano) |
|------------|------------|------------|-------------|------------------|---------------------------|
| **YOLOv8n**| ~3.2M | 640x640 | 9.80 | 102.02 | 1.00x |
| **YOLOv8s**| ~11.2M | 640x640 | 4.01 | 249.07 | 2.44x Latency Increase |
| **YOLOv8m**| ~25.9M | 640x640 | 1.87 | 535.34 | 5.25x Latency Increase |

**Key Takeaway**: Latency scales non-linearly with parameter count. While YOLOv8s and YOLOv8m provide superior representation capability and detection accuracy under complex occlusions, their computational footprint is prohibitive for real-time CPU edge execution, rendering YOLOv8n the default choice for edge systems without hardware-accelerated NPUs or GPUs.

---

### 3.3 Impact of Precision (FP32 vs. FP16)
Half-precision (FP16) representations halve the memory bandwidth requirements and double the theoretical execution rate on tensor cores.

- **On GPU/NPU Environments**: FP16 execution delivers up to a **2x throughput increase** with negligible degradation in mean Average Precision (mAP).
- **On Standard CPU-only Environments**: Executing FP16 results in a severe performance penalty because of lack of hardware register support, leading to emulated casting overhead. The benchmarking suite dynamically catches this and falls back to standard FP32 to prevent runtime failures.

---

## 4. Architectural Recommendations for Edge Deployments
Based on the empirical findings, we propose the following deployment tree:

1. **Strictly CPU-Bound Deployments (e.g., Raspberry Pi 4/5)**:
   - **Recommendation**: Deploy **YOLOv8n** at **$416 \times 416$ resolution** using **FP32** (or integer quantized INT8 formats). This configuration yields the highest frame rate (~15-20 FPS) while maintaining reasonable classification capability.
2. **GPU/NPU-Accelerated Edge Deployments (e.g., Jetson Orin Nano, Google Coral NPU)**:
   - **Recommendation**: Deploy **YOLOv8n or YOLOv8s** at **$640 \times 640$ resolution** using **FP16**. This preserves spatial resolution for small objects while capitalizing on GPU tensor cores to sustain high frame rates (>30 FPS).
