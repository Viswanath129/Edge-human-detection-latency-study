## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution, model architecture, and inference precision affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a standardized YOLO-based human detection pipeline and benchmark performance across multiple configurations. The benchmarking suite handles:
- **Synthetic Fallback:** Ensures reliable testing in headless environments.
- **Warmup Phase:** Minimizes initial cold-start variance.
- **Inference-only Timing:** Provides technically precise performance metrics by excluding I/O overhead.

## Experiments
- **Resolution comparison:** 640×640 vs 416×416
- **Model Size comparison:** YOLOv8 Nano vs YOLOv8 Small
- **Precision comparison:** FP32 vs FP16 (requires CUDA)

## Usage
To run the full benchmark suite:
```bash
export FORCE_SYNTHETIC=true
python3 experiments/run_all.py
```

## Results & Analysis
Detailed findings and comparative plots are available in the `results/` directory and documented in `report/research_note.md`.

## Preliminary Results (Typical)
| Resolution | Model | Precision | FPS | Avg Latency (ms) | Notes |
|------------|-------|-----------|-----|------------------|-------|
| 640x640    | Nano  | FP32      | ~8  | ~110             | High quality |
| 416x416    | Nano  | FP32      | ~14 | ~65              | Balanced |
| 640x640    | Small | FP32      | ~3  | ~300             | Accurate but slow |

## Limitations
- CPU-only benchmarking in most environments.
- Accuracy estimation is qualitative in this phase.

## Future Work
- INT8 quantization for Edge NPUs.
- Multi-thread batch inference analysis.
