import os
from utils import benchmark_model, save_summary

def run_resolution_test():
    model_path = "yolov8n.pt"
    resolutions = [640, 416]
    force_synthetic = os.getenv("FORCE_SYNTHETIC", "false").lower() == "true"

    print(f"Starting resolution comparison for YOLOv8n...")

    for imgsz in resolutions:
        resolution_str = f"{imgsz}x{imgsz}"
        print(f"Benchmarking {resolution_str}...")

        avg_latency, fps, _ = benchmark_model(
            model_path,
            imgsz=imgsz,
            force_synthetic=force_synthetic
        )

        observation = "Higher detection quality" if imgsz == 640 else "Faster Inference"

        save_summary(
            resolution=resolution_str,
            model_name="yolov8n",
            precision="FP32",
            fps=fps,
            latency=avg_latency,
            observation=observation
        )

if __name__ == "__main__":
    run_resolution_test()
