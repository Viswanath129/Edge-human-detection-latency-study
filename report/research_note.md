# Research Note: Latency-Accuracy Trade-offs in Edge Human Detection

## 1. Experimental Setup
All experiments were conducted using the YOLOv8 architecture (Nano and Small variants).
- **Environment**: Synthetic frame generation (`FORCE_SYNTHETIC=true`)
- **Hardware**: CPU-based inference
- **Software**: Python 3.x, Ultralytics YOLOv8, OpenCV, Torch

## 2. Key Findings

### 2.1 Resolution Impact
Reducing the input resolution from 640px to 416px significantly reduces the computational load. For YOLOv8n, we observed an increase from ~8 FPS to ~18 FPS, making it more suitable for real-time applications on low-power devices.

### 2.2 Model Complexity
The jump from YOLOv8n (nano) to YOLOv8s (small) more than doubles the average latency (115ms to 257ms). While 'small' models generally provide better detection of small objects, the latency cost is substantial for edge deployment.

### 2.3 Precision and Hardware Acceleration
Half-precision (FP16) tests on CPU showed extremely poor performance (~0.14 FPS). This confirms that FP16 is only viable when using hardware specifically optimized for it, such as NVIDIA CUDA cores or dedicated NPUs.

## 3. Visualizations
Visualizations for these comparisons can be found in `results/plots/`:
- `latency_vs_resolution.png`
- `latency_vs_model_size.png`
- `latency_vs_precision.png`
