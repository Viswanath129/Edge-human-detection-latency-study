import os
import torch
from utils import run_benchmark, save_results

def main():
    headless = os.environ.get("HEADLESS", "false").lower() == "true"

    # FP32 is always tested. FP16 is tested regardless of CUDA for research purposes,
    # though notes on performance are provided in README.
    precisions = [False, True]

    if torch.cuda.is_available():
        print("CUDA available, testing FP32 and FP16.")
    else:
        print("CUDA not available. FP16 may be slow on CPU, but benchmarking for research.")

    summary_data = []
    all_raw_latencies = {}

    for half in precisions:
        precision_name = "FP16" if half else "FP32"
        print(f"Benchmarking precision: {precision_name}")

        avg_latency, fps, latencies = run_benchmark("yolov8n.pt", half=half, headless=headless)

        summary_data.append({
            "precision": precision_name,
            "avg_latency_ms": avg_latency,
            "fps": fps
        })
        all_raw_latencies[precision_name] = latencies
        print(f"{precision_name} - Avg Latency: {avg_latency:.2f} ms, FPS: {fps:.2f}")

    save_results("precision_results.csv", summary_data, all_raw_latencies)

if __name__ == "__main__":
    main()
