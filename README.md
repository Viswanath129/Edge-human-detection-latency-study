## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices. We analyze how input resolution, model architecture, and numerical precision affect inference speed and resource utilization.

## Research Question
How do input resolution, model configuration, and precision levels affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLOv8-based human detection pipeline with a centralized benchmarking utility to measure:
- **Inference Latency:** Time taken per frame for model prediction.
- **Inference FPS:** Throughput based on inference-only timing.
- **Comparative Analysis:** Evaluation across multiple resolutions, model sizes, and precision modes.

## Performance Analysis Suite
The repository includes automated scripts to benchmark different configurations:
- `resolution_test.py`: Compares 640x640 vs 416x416 input sizes.
- `model_size_test.py`: Compares YOLOv8n (Nano) vs YOLOv8s (Small) variants.
- `precision_test.py`: Compares FP32 vs FP16 precision.

### Running Benchmarks
To run the full suite and generate plots:
```bash
export FORCE_SYNTHETIC=true  # Use synthetic frames if no webcam is available
python3 experiments/run_all.py
```

## Preliminary Results Summary
| Resolution | Model | Precision | FPS | Avg Latency (ms) |
|------------|-------|-----------|-----|------------------|
| 640x640    | Nano  | FP32      | ~10 | ~100             |
| 416x416    | Nano  | FP32      | ~20 | ~50              |
| 640x640    | Small | FP32      | ~4  | ~230             |

## Observations
- **Resolution Impact:** Lowering resolution from 640 to 416 roughly doubles the FPS.
- **Model Size:** The Small variant is ~2.3x slower than the Nano variant on standard CPUs.
- **Bottlenecks:** CPU-only inference remains the primary bottleneck; NPU/GPU acceleration is recommended for the "Small" model variant.

## Future Work
- Edge NPU benchmarking (Jetson, Coral, etc.)
- INT8 Quantization impact analysis
- Multi-object tracking latency overhead evaluation
