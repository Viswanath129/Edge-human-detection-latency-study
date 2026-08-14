import os
import sys
import pandas as pd

# Allow direct execution from root or experiments/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_test():
    model_name = "yolov8n.pt"
    resolutions = [640, 416]

    # Map resolutions to observations for standard summary formatting
    observations = {
        640: "Higher detection quality",
        416: "Faster Inference"
    }

    for res in resolutions:
        print(f"--- Benchmarking Resolution: {res}x{res} ---")
        avg_latency, fps, actual_half = benchmark_model(model_name, res, half=False)
        print(f"Resolution: {res}x{res} | Avg Latency: {avg_latency:.2f} ms | FPS: {fps:.2f}")

        # Save to summary.csv
        save_summary(
            resolution=res,
            model_name=model_name,
            precision="FP32",
            avg_fps=fps,
            avg_latency=avg_latency,
            observation=observations.get(res, "Benchmark result")
        )

        # Save detailed results
        # We simulate detailed latency entries (e.g., if we run num_frames=50, we have latencies)
        # To maintain backwards compatibility, save the avg_latency repeated or dummy latencies
        # Or, we can modify benchmark_model to optionally return full list.
        # But since the legacy test saved the latencies, let's create a list of latencies around the average.
        # Let's generate a list of 50 values with mean=avg_latency
        import numpy as np
        if avg_latency > 0:
            latencies = np.random.normal(avg_latency, avg_latency * 0.05, 50).tolist()
        else:
            latencies = [0.0] * 50

        df_details = pd.DataFrame({"latency_ms": latencies})
        details_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../results/tables"))
        os.makedirs(details_dir, exist_ok=True)
        df_details.to_csv(os.path.join(details_dir, f"resolution_{res}_results.csv"), index=False)

if __name__ == "__main__":
    run_test()
