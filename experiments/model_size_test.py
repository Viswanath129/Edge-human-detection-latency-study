import sys
import os

# Add the current directory to sys.path to allow importing utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_model_size_tests():
    models = ["yolov8n.pt", "yolov8s.pt"]
    resolution = 640

    print(f"Starting model size tests at {resolution}x{resolution}...")

    for model_path in models:
        model_name = "YOLOv8n" if "n.pt" in model_path else "YOLOv8s"
        print(f"Testing model: {model_name}")

        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=resolution)

        precision = "FP16" if actual_half else "FP32"
        observation = "Nano model (fastest)" if "n.pt" in model_path else "Small model (more accurate)"

        save_summary(
            resolution=f"{resolution}x{resolution}",
            model_name=model_name,
            precision=precision,
            fps=fps,
            latency=avg_latency,
            observation=observation
        )
        print(f"Model {model_name} complete: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_model_size_tests()
