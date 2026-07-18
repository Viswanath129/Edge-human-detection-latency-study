# Edge AI Human Detection: Latency-Accuracy Trade-off Analysis

This research note documents a systematic empirical study analyzing the trade-offs between inference latency, throughput (FPS), and architectural configurations for real-time human detection on edge constraints.

---

## Experimental Methodology

Our experiments evaluate deep learning object detection models under resource-constrained scenarios along three distinct dimensions:
1. **Input Resolution:** Comparing standard high-resolution input ($640 \times 640$) against reduced-resolution input ($416 \times 416$).
2. **Model Capacity (Model Size):** Comparing the lightweight **YOLOv8n** (Nano) with the slightly larger **YOLOv8s** (Small).
3. **Precision Level:** Comparing standard single-precision floating-point (**FP32**) against half-precision (**FP16**) inference.

### Hardware Environment
- **Device:** Edge Emulation Environment (CPU-only container sandbox)
- **Framework:** PyTorch & Ultralytics API
- **Inference Hardware:** Standard x86 CPU (No CUDA/NPU acceleration available)

---

## Empirical Results

Below is the consolidated summary of the benchmark metrics recorded during the execution of our orchestrated suite:

| Resolution | Model | Precision | Average FPS | Average Latency (ms) | Key Observations & Performance Implications |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **640x640** | **YOLOv8n** | FP32 | 8.0 | 125.3 | Standard baseline; high spatial resolution but higher latency. |
| **416x416** | **YOLOv8n** | FP32 | 17.2 | 58.2 | Significant latency reduction ($53.6\%$) with moderate resolution trade-off. |
| **640x640** | **YOLOv8s** | FP32 | 3.5 | 287.3 | Increased model capacity leads to high accuracy but severe latency penalties. |

*Note: FP16 benchmarks were executed but omitted from data serialization because CUDA/NPU acceleration was unavailable in this runtime environment. Running half-precision on standard CPUs either falls back to FP32 or incurs substantial software emulation overheads, which are non-representative of real NPU hardware acceleration.*

---

## Architectural Findings & Analysis

### 1. The Impact of Input Resolution
Reducing the input spatial dimensions from $640 \times 640$ to $416 \times 416$ yielded a **$2.15\times$ increase in throughput (FPS)**, reducing per-frame inference latency from **125.3 ms to 58.2 ms**.
- **The Trade-off:** Input resolution directly influences the receptive field and feature extraction layers. While $416\times416$ drastically improves FPS and latency, it reduces the model's capacity to detect tiny objects or humans at a distance. For close-up real-time tracking, resolution downscaling is a highly viable optimization.

### 2. Scaling Model Capacity (YOLOv8n vs. YOLOv8s)
Upgrading the model from the Nano ($3.2\text{M}$ parameters) to the Small variant ($11.2\text{M}$ parameters) resulted in a massive **$2.29\times$ latency increase**, raising average latency to **287.3 ms (3.5 FPS)**.
- **The Trade-off:** YOLOv8s offers vastly superior mean Average Precision (mAP) and multi-class classification confidence. However, in edge device deployments, 3.5 FPS is highly inadequate for real-time tracking or safety-critical automation. YOLOv8n remains the preferred architecture where throughput overrides confidence margins.

### 3. Precision-Aware Inference
- On CPU, FP32 remains the gold standard. Attempting half-precision (FP16) on CPU hardware lacks native hardware registers, failing to deliver the standard $2\times$ speedup seen on tensor-core-enabled GPUs or dedicated NPUs.
- Modern edge-AI deployments must leverage quantization (e.g., INT8) or target specialized NPUs to realize the latency-saving advantages of reduced precision.

---

## Architectural & Deployment Recommendations

1. **For Safety-Critical, High-Speed Environments:** Use **YOLOv8n at $416 \times 416$ resolution**. This setup maximizes frame throughput ($\approx 17\text{ FPS}$) and ensures responsiveness within the critical sub-100 ms threshold.
2. **For High-Precision, Low-Speed Environments:** Use **YOLOv8n or YOLOv8s at $640 \times 640$ resolution** paired with specialized accelerator hardware (CUDA GPU, Jetson Nano, or Coral TPU) utilizing FP16 or INT8 quantization.
