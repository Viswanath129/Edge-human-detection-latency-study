import os
import sys

# Insert containing directory to sys.path to allow running from repository root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    print("Starting Resolution Benchmark Test...")
    model_name = "yolov8n.pt"

    # We test two resolutions: 640 and 416
    resolutions = [640, 416]
    observations = {
        640: "Higher detection quality",
        416: "Faster Inference"
    }

    for res in resolutions:
        print(f"Benchmarking {model_name} at {res}x{res}...")
        avg_latency, fps, actual_half = benchmark_model(model_name, res, half=False)

        print(f"Result - Avg Latency: {avg_latency:.2f} ms, FPS: {fps:.2f}")

        # Save summary to results/tables/summary.csv
        save_summary(
            resolution=res,
            model_name=model_name,
            precision="FP32",
            avg_fps=fps,
            avg_latency=avg_latency,
            observation=observations[res]
        )

    print("Resolution Benchmark Test completed and results saved.")

if __name__ == "__main__":
    main()
