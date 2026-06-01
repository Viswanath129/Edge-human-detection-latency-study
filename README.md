# Edge Human Detection: Latency-Accuracy Trade-offs

## Project Overview
This research evaluates the performance of YOLOv8-based human detection on edge compute environments, focusing on the trade-offs between input resolution, model architecture, and numerical precision.

## Research Questions
1. How does input resolution scaling affect inference throughput and latency?
2. What is the performance penalty of moving from nano-scale to small-scale models?
3. Can half-precision (FP16) inference provide meaningful speedups in edge environments?

## Methodology
Benchmarks are conducted using a standardized pipeline in `experiments/utils.py`. The suite measures inference-only latency over 20 frames after a 5-frame warmup phase to ensure stable metrics.

### Setup
- **Model Family**: Ultralytics YOLOv8
- **Hardware Simulation**: Headless environment with synthetic frame fallback
- **Metrics**: Average FPS, Average Latency (ms)

## Experimental Results

### 1. Resolution Scaling (YOLOv8n, FP32)
| Resolution | FPS | Latency (ms) | Observation |
|------------|-----|--------------|-------------|
| 640x640    | 7.2 | 138.6        | High fidelity, suitable for distant detection |
| 416x416    | 15.2| 65.9         | ~2x speedup, ideal for fast motion |

### 2. Model Architecture (640x640, FP32)
| Model | Variant | FPS | Latency (ms) |
|-------|---------|-----|--------------|
| YOLOv8n | Nano | 7.2 | 138.6 |
| YOLOv8s | Small | 3.2 | 316.8 |

### 3. Precision Analysis (YOLOv8n, 640x640)
| Precision | FPS | Latency (ms) | Notes |
|-----------|-----|--------------|-------|
| FP32      | 7.1 | 141.6        | Standard baseline |
| FP16      | 8.0 | 125.2        | Significant speedup on compatible hardware |

## Key Observations
- **Resolution is the primary lever**: Reducing resolution from 640 to 416 provides a near-linear speedup in inference throughput.
- **Model Scaling Costs**: Transitioning from Nano to Small increases latency by ~2.3x, which may exceed real-time requirements for many edge devices.
- **Precision Benefits**: FP16 inference shows promise for reducing latency, though actual gains are hardware-dependent (optimized for CUDA/NPU).

## Limitations
- Benchmarks conducted on general-purpose CPU; results will vary significantly on dedicated edge NPUs (e.g., Jetson, Coral).
- Accuracy is not quantitatively measured in this phase; observations are based on qualitative architectural expectations.

## Reproducing Results
Run the full suite:
```bash
python3 experiments/run_all.py
python3 results/plots/plot_results.py
```
