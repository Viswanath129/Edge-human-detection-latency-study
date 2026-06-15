## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution and model configuration affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency during live video inference.

## Summary of Results
| Model | Resolution | Precision | FPS | Avg Latency (ms) |
|-------|------------|-----------|-----|------------------|
| YOLOv8n | 416x416 | FP32 | 21.1 | 47.5 |
| YOLOv8n | 640x640 | FP32 | 9.7 | 102.9 |
| YOLOv8s | 640x640 | FP32 | 4.4 | 225.9 |

## Observations
- **Resolution**: Reducing resolution to 416x416 more than doubles the FPS for the Nano model.
- **Model Size**: YOLOv8n is significantly faster than YOLOv8s, making it ideal for low-power edge devices.
- **Precision**: Hardware acceleration is critical for seeing benefits from FP16 precision.

## Visualizations
Generated plots can be found in `results/plots/`:
- `latency_vs_resolution.png`
- `fps_vs_resolution.png`
- `fps_vs_model.png`
- `latency_precision_comp.png`

## Detailed Analysis
For a deep dive into the experimental findings, see [report/research_note.md](report/research_note.md).

## Future Work
- Edge NPU benchmarking
- INT8 quantization
- Multi-object detection performance analysis
