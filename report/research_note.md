# Latency–Accuracy Trade-off Analysis

## Experimental Overview
This research explores the impact of three key variables on real-time human detection for edge devices using the YOLOv8 architecture:
1. **Input Resolution:** 640x640 vs 416x416.
2. **Model Architecture:** YOLOv8n (Nano) vs YOLOv8s (Small).
3. **Precision:** FP32 vs FP16 (Hardware dependent).

## Key Findings

### 1. Resolution Impact
Lowering the input resolution from 640x640 to 416x416 results in a significant reduction in inference latency (~40%). While 640x640 provides better accuracy for small objects, 416x416 is highly effective for high-throughput edge requirements.

### 2. Model Architecture Trade-offs
- **YOLOv8n:** Optimized for extreme edge constraints, providing the highest FPS with minimal memory overhead.
- **YOLOv8s:** Offers better detection confidence at the cost of increased latency (approximately double that of YOLOv8n on CPU).

### 3. Precision Optimization
FP16 precision offers significant acceleration on CUDA-enabled hardware without substantial accuracy loss. On standard CPU environments, FP32 remains the most stable option.

## Conclusion
For real-time edge human detection, YOLOv8n at 416x416 resolution represents the optimal balance of speed and detection reliability. If hardware acceleration (CUDA/NPU) is available, enabling FP16 and utilizing YOLOv8s at 640x640 can provide superior accuracy while maintaining real-time performance.
