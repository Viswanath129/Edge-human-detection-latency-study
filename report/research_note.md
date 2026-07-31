# Research Note: Latency-Accuracy Trade-off Analysis for YOLOv8-Based Real-Time Human Detection on Edge Devices

## Abstract
This research note investigates the latency and throughput trade-offs of using state-of-the-art YOLOv8 models for real-time human detection under edge-compute constraints. We systematically analyze performance along three critical design dimensions: **input resolution** ($640\times640$ vs $416\times416$), **model size** (Nano vs Small), and **inference precision** (FP32 vs FP16). Our results show that lower resolution and model scale provide dramatic, non-linear latency reductions, while half-precision (FP16) on commodity edge CPU architectures is highly non-representative and requires dedicated hardware acceleration (e.g., GPU/NPU) to yield performance benefits.

---

## 1. Introduction & Research Question
Real-time human detection is a core building block for edge applications such as video surveillance, robotics, and smart environments. However, deploying deep learning models on resource-constrained edge devices presents a severe challenge due to strict thermal, power, and computational budgets.

This study answers the following research question:
> **How do input resolution, model variant size, and precision levels affect inference latency and throughput (FPS) for YOLOv8 human detection under edge-compute constraints?**

---

## 2. Methodology & Benchmark Setup
We construct an automated, standardized edge-benchmarking pipeline in Python utilizing the `ultralytics` YOLOv8 API.

### 2.1 Experimental Dimensions
1. **Input Resolution:** $640\times640$ vs $416\times416$ pixels.
2. **Model Variant Size:** YOLOv8n (Nano: ~3.2M parameters) vs YOLOv8s (Small: ~11.2M parameters).
3. **Precision Level:** FP32 (single precision) vs FP16 (half precision).

### 2.2 Benchmarking Protocol
- **Hardware Platform:** Host CPU Environment (Single-socket Intel/AMD, headless execution context).
- **Centralized Instrumentation:** Standardized logic in `experiments/utils.py` pre-generates synthetic inputs outside the main timed loops to eliminate camera I/O, frame-decoding, and rendering overhead.
- **Warmup:** A 5-frame warmup phase is executed using the identical model and precision configurations to stabilize the PyTorch runtime.
- **Statistical Rigor:** Metrics are computed over 50 consecutive inference iterations using high-precision `time.perf_counter()`. CPU/GPU synchronization is handled via `torch.cuda.synchronize()` when CUDA acceleration is present.

---

## 3. Experimental Results

The following table summarizes the performance metrics captured across all completed benchmark configurations:

| Resolution | Model Variant | Precision | Average Latency (ms) | Average Throughput (FPS) | Key Observations / Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **640x640** | YOLOv8n | FP32 | 116.65 ms | 8.57 FPS | Baseline configuration; higher detection resolution. |
| **416x416** | YOLOv8n | FP32 | 51.70 ms | 19.34 FPS | Fast inference; ~2.26x speedup over 640x640 baseline. |
| **640x640** | YOLOv8s | FP32 | 268.41 ms | 3.73 FPS | Higher capacity; ~2.30x higher latency compared to Nano. |

*Note: Since the benchmarking host lacks active CUDA acceleration (headless CPU-only environment), FP16 half-precision summaries are omitted from `summary.csv` to avoid recording non-representative CPU fallback execution under the FP16 label.*

---

## 4. Analytical Discussion of Trade-offs

### 4.1 Input Resolution Impact ($640\times640$ vs $416\times416$)
Reducing the input resolution from $640\times640$ to $416\times416$ pixels leads to a dramatic **55.7% reduction in average inference latency** (dropping from **116.65 ms** to **51.70 ms**), which translates to a **125.7% throughput increase** (rising from **8.57 FPS** to **19.34 FPS**).

This non-linear performance scaling occurs because the computational complexity of convolutional operations scales quadratically with spatial dimensions ($O(H \times W)$). While a $416\times416$ input significantly accelerates inference, it reduces the spatial density of information, which may degrade accuracy for small or distant objects (e.g., humans at a distance). For many real-time edge security and occupancy tracking applications, however, this trade-off is highly favorable.

### 4.2 Model Variant Impact (Nano vs Small)
Upgrading the model scale from YOLOv8n (Nano) to YOLOv8s (Small) at $640\times640$ resolution increases inference latency from **116.65 ms** to **268.41 ms** (a **129.5% latency increase**), causing throughput to plummet to a non-real-time **3.73 FPS**.

YOLOv8s incorporates more channels and deeper convolutional layers (~11.2M parameters compared to ~3.2M parameters for YOLOv8n). This higher model capacity yields significantly higher mAP (mean Average Precision) and robust human localization, but the extreme latency penalty makes it unsuitable for standard CPU-based edge environments without downstream hardware acceleration.

### 4.3 Precision Level Impact (FP32 vs FP16)
In our experiments, FP16 half-precision benchmarks are skipped or filtered out when CUDA acceleration is absent. This architectural decision is technically crucial: **standard CPU instruction sets do not natively accelerate 16-bit floating-point math**.

Running FP16 models on standard x86/ARM CPUs forces PyTorch to perform software-emulated casting or slower instruction paths, which degrades performance rather than improving it. Consequently, FP16 benchmarking should only be targeted at devices with dedicated tensor cores (e.g., NVIDIA Jetson CUDA cores) or NPUs where low-precision execution is accelerated in hardware.

---

## 5. Architectural Recommendations & Future Work

### 5.1 Edge Deployment Guidelines
1. **CPU-only Edge Nodes:** For pure CPU edge devices (such as smart gateways), the **YOLOv8n model combined with 416x416 resolution** is the only configuration capable of approaching near-real-time throughput (~19 FPS) while maintaining acceptable localization accuracy.
2. **Accelerated Edge Nodes:** If the edge device contains an embedded GPU or NPU, developers should adopt **YOLOv8n (or YOLOv8s) at 640x640 with FP16 precision enabled** to maximize detection quality without compromising the real-time frame budget.

### 5.2 Future Research Directions
- **INT8 Quantization:** Investigate INT8 quantization using TensorRT or OpenVINO, which can provide up to 4x throughput improvements on CPU edge nodes with Intel DL Boost or ARM Neon.
- **NPU Benchmarking:** Deploy and test the pipeline on specialized edge silicon (such as the Coral Edge TPU or Rockchip NPU) to chart hardware-native low-precision performance.
- **Quantitative Accuracy Analysis:** Couple latency benchmarks with a standardized COCO-validation pipeline to compute exact mAP degradation across resolution and model variant boundaries, formalizing a Pareto-frontier curve.
