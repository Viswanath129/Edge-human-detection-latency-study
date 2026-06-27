# Experimental Findings: YOLO-based Human Detection on Edge Devices

## 1. Impact of Input Resolution
Reducing input resolution from 640x640 to 416x416 significantly decreases inference latency.
- **640x640**: Higher detail, better for small-scale objects.
- **416x416**: Roughly 40-50% faster, suitable for high-speed tracking where objects are large or close to the camera.

## 2. Model Complexity Trade-offs
The jump from YOLOv8 Nano to Small doubles the latency but offers improved mAP. On extremely resource-constrained edge CPUs, Nano is the only viable option for real-time ( > 10 FPS) performance without hardware acceleration.

## 3. Precision-Aware Inference
FP16 (Half Precision) is primarily beneficial on NPU/GPU hardware. On standard ARM or x86 CPUs without specialized vector instructions, FP16 might actually be slower due to software-level emulation.

## Summary Table
See `results/tables/summary.csv` for detailed metrics across all tested configurations.
