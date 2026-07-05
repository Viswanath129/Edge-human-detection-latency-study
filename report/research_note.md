# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Experimental Setup
- **Hardware**: CPU-based benchmarking (Synthetic frames fallback for headless environment)
- **Model**: YOLOv8 (Nano and Small variants)
- **Task**: Human detection (Inference only)

## Performance Metrics

| Resolution | Model | Precision | Avg FPS | Avg Latency (ms) |
|------------|-------|-----------|---------|------------------|
| 640x640    | Nano  | FP32      | 8.40    | 119.10           |
| 416x416    | Nano  | FP32      | 19.56   | 51.12            |
| 640x640    | Small | FP32      | 3.94    | 253.53           |

## Key Findings

### 1. Impact of Input Resolution
Reducing the input resolution from 640x640 to 416x416 resulted in a **2.3x increase in throughput** (from 8.4 to 19.56 FPS). This confirms that input scaling is the most effective lever for achieving real-time performance on resource-constrained edge CPUs.

### 2. Model Architecture Complexity
The transition from the Nano (n) to the Small (s) model at 640x640 resolution resulted in a **2.1x increase in latency**. While the Small model offers higher detection accuracy (mAP), the performance cost is significant, pushing the inference speed well below real-time thresholds for most edge applications.

### 3. Precision and Hardware Acceleration
Tests conducted on CPU environments show that FP32 is the standard for general-purpose inference. While FP16 logic was implemented, it was bypassed in this suite due to the lack of CUDA acceleration. On standard CPUs, FP16 typically does not offer performance gains and can sometimes be slower due to software-level emulation.

## Conclusion
For real-time human detection on edge devices without dedicated NPUs or GPUs, the **YOLOv8n model at 416x416 resolution** provides the most viable performance profile (~20 FPS). Further optimizations such as INT8 quantization should be explored for even lower latency requirements.
