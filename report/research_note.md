# Research Note: Latency-Accuracy Trade-off Analysis for Real-Time Human Detection on Edge Devices

## Executive Summary
This research note presents an empirical study evaluating the trade-offs between inference speed, latency, and model configuration for YOLO-based human detection models deployed under edge compute constraints. Our benchmarks cover model variations, input resolutions, and precision levels to provide engineering recommendations for deploying real-time vision pipelines on constrained systems.

---

## 1. Research Question
*How do input resolution, model variant size, and precision levels affect inference latency and detection throughput (FPS) for YOLOv8 human detection under edge-representative constraints?*

---

## 2. Methodology
Our automated benchmarking pipeline isolates inference performance using standardized parameters. To guarantee scientific repeatability, the methodology incorporates:
- **Frame Source Isolation**: The benchmarking module utilizes webcam capture or falls back to synthetic input frame generation (pre-generated outside the timed inference loop) to eliminate frame-acquisition/disk-I/O overhead.
- **Warmup Controls**: A 5-frame warmup phase is executed using the exact same configuration as the main benchmark to stabilize model compilation/caching states.
- **Timing Precision**: Benchmarking calculations use high-resolution CPU performance counters (`time.perf_counter()`) and explicit device synchronizations (`torch.cuda.synchronize()`) when hardware acceleration is present.
- **Evaluation Settings**:
  - **Models**: YOLOv8 Nano (`yolov8n`), YOLOv8 Small (`yolov8s`)
  - **Resolutions**: $640 \times 640$, $416 \times 416$
  - **Precision Levels**: FP32 (Single-precision float), FP16 (Half-precision float)

---

## 3. Experimental Results

The table below summarizes the recorded performance metrics under a CPU-only execution environment:

| Resolution | Model Size | Precision | Average Latency (ms) | Average FPS | Key Observation |
|------------|------------|-----------|----------------------|-------------|-----------------|
| $640 \times 640$ | YOLOv8n (Nano) | FP32 | 109.46 | 9.14 | Baseline optimized nano model |
| $416 \times 416$ | YOLOv8n (Nano) | FP32 | 51.85 | 19.29 | Significantly faster; ideal for high throughput |
| $640 \times 640$ | YOLOv8s (Small) | FP32 | 256.53 | 3.90 | Higher capacity but poor real-time feasibility on CPU |

---

## 4. Key Insights & Discussion

### 4.1 Resolution vs. Latency
Lowering the input resolution from $640 \times 640$ to $416 \times 416$ leads to a **~52.6% reduction in latency** (from $109.46\text{ ms}$ to $51.85\text{ ms}$), corresponding to an increase in frame rate from **$9.14\text{ FPS}$ to $19.29\text{ FPS}$**. On edge devices without dedicated hardware accelerators, reducing the input resolution is the most effective lever to achieve near real-time throughput.

### 4.2 Model Size Impact
Upgrading from YOLOv8n to YOLOv8s at $640 \times 640$ input size increases the average per-frame latency from **$109.46\text{ ms}$ to $256.53\text{ ms}$** (a **$2.34\times$ increase**). This represents a severe degradation in frame rate ($9.14\text{ FPS}$ down to $3.90\text{ FPS}$), rendering the small model unfeasible for real-time edge streaming applications unless paired with dedicated acceleration (e.g., NPU or TensorRT engine).

### 4.3 Precision Scaling (FP32 vs. FP16)
Half-precision (FP16) benchmarking is mathematically designed for hardware-accelerated instructions (such as CUDA Tensor Cores or NPUs). Running FP16 inference on standard non-accelerated CPU hardware does not provide performance improvements and can lead to slower/non-representative performance or fallback to FP32.

---

## 5. Limitations & Risks
1. **CPU Execution Limitations**: Current edge benchmarks are executed on host-simulated CPU environments. True physical NPU/GPU execution characteristics (such as memory bandwith bottle-necks) may diverge.
2. **Missing Real Accuracy Metrics**: This study focuses strictly on latency and FPS metrics; trade-off curves do not incorporate formal mAP (Mean Average Precision) values on target edge datasets (e.g., COCO-person).
3. **Synthetic Frame Variance**: Although synthetic zero-frames guarantee minimal frame acquisition noise, actual live imagery can exhibit slight variations in post-processing (e.g., NMS overhead).
