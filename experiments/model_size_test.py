import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_model_size_test():
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for model_path in models:
        model_name = model_path.split(".")[0]
        print(f"\n--- Testing Model: {model_name} ---")

        avg_latency, fps, is_half = benchmark_model(model_path, imgsz=res)

        precision = "FP16" if is_half else "FP32"
        obs = "Nano (Fastest)" if "nano" in model_name or model_name.endswith('n') else "Small (Better accuracy)"

        save_summary(
            resolution=f"{res}x{res}",
            model_name=model_name,
            precision=precision,
            fps=fps,
            latency=avg_latency,
            observation=obs
        )

if __name__ == "__main__":
    run_model_size_test()
