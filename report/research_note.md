# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## Experimental Setup
This study evaluates YOLOv8 models under various configurations to identify optimal deployment parameters for edge devices. Experiments were conducted using synthetic frame fallback to ensure consistency across environments.

## 1. Input Resolution Impact
Reducing input resolution from 640x640 to 416x416 significantly impacts throughput.

- **640x640**: Higher spatial resolution allows for better detection of small or distant subjects but at a higher computational cost.
- **416x416**: Provides a substantial boost in FPS (approx. 2x) with a corresponding reduction in per-frame latency. This is often the preferred choice for real-time applications where motion blur or high-speed tracking is required.

## 2. Model Architecture Comparison
We compared YOLOv8 Nano (n) and Small (s) variants.

- **YOLOv8n**: Optimized for edge deployment. It maintains a high frame rate but may struggle with occlusion or complex backgrounds.
- **YOLOv8s**: Offers improved Mean Average Precision (mAP) but results in significantly higher latency (often double that of the nano version).

## 3. Precision-Aware Inference
FP16 (Half Precision) was evaluated as a potential optimization.

- **Findings**: On hardware supporting FP16 (e.g., modern GPUs/NPUs), half-precision can accelerate inference without significant accuracy loss. On standard CPUs, FP32 remains the baseline as FP16 may fallback or even perform slower due to lack of specialized instructions.

## Conclusion
For real-time human detection on edge devices, **YOLOv8n at 416x416 resolution** provides the most viable balance between responsiveness and detection capability.
