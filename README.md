## Project Overview
This project studies the latency–accuracy trade-offs of real-time human detection models deployed on edge devices.

## Research Question
How do input resolution and model configuration affect inference latency and detection accuracy under edge compute constraints?

## Methodology
We implement a real-time YOLO-based human detection pipeline and benchmark FPS and per-frame latency during live video inference.

## Experiments
- Resolution comparison (640×640 vs 416×416)

## Preliminary Results
| Resolution | FPS | Avg Latency (ms) | Notes |
|------------|-----|------------------|-------|
| 640        | 7.6  | 110.0               | Higher accuracy |
| 416        | 14.2  | 65.0               | Faster inference |

## Observations
- Lower input resolution significantly improves inference latency.
- Accuracy degradation is moderate for human detection tasks.

## Limitations
- Limited dataset size
- Approximate accuracy estimation
- Single-device benchmarking

## Future Work
- Precision-aware inference (FP16 / INT8)
- Edge NPU benchmarking
- Model size comparison
