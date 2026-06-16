# Experimental Results: Edge Human Detection

## 1. Resolution Trade-offs
Reducing input resolution from 640x640 to 416x416 significantly decreases inference latency, making it more suitable for real-time applications on resource-constrained devices, albeit with a slight reduction in detection granularity for small objects.

## 2. Model Architecture
YOLOv8n (Nano) offers the best latency-performance ratio for edge deployment. Stepping up to YOLOv8s (Small) increases accuracy but results in higher per-frame latency.

## 3. Precision Optimization
Floating-point 16 (FP16) inference can provide substantial speedups on compatible hardware (NPU/GPU) without significant loss in detection accuracy compared to standard FP32.

## Key Observations
- Inference-only FPS is the primary metric for evaluating model throughput.
- Cold-start latency is mitigated by a 5-frame warmup phase in our benchmarking suite.
- Synthetic fallback ensures benchmarking reliability in headless or non-camera environments.
