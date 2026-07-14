import os
from utils import benchmark_model, save_summary

def run_model_size_test():
    models = ["yolov8n.pt", "yolov8s.pt"]
    resolution = "640x640"
    imgsz = 640
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"

    print(f"Starting model size comparison at {resolution}...")

    for model_path in models:
        model_name = os.path.basename(model_path).replace(".pt", "")
        print(f"Benchmarking {model_name}...")

        avg_latency, fps, _ = benchmark_model(
            model_path,
            imgsz=imgsz,
            force_synthetic=force_synthetic
        )

        observation = "Standard nano model" if "n" in model_name else "Larger model, higher accuracy"

        save_summary(
            resolution=resolution,
            model_name=model_name,
            precision="FP32",
            fps=fps,
            latency=avg_latency,
            observation=observation
        )

if __name__ == "__main__":
    run_model_size_test()
