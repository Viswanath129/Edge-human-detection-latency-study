import os
import sys

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_model_size_tests():
    models = ["yolov8n.pt", "yolov8s.pt"]
    resolution = 640

    for model_name in models:
        # Extract model tag (e.g., yolov8n)
        model_tag = model_name.split('.')[0]
        print(f"\n--- Testing Model Size: {model_tag} ---")

        avg_latency, fps, _ = benchmark_model(model_name, resolution)

        observation = "Ultra-lightweight" if "n" in model_tag else "Balanced performance"
        save_summary(f"{resolution}x{resolution}", model_tag, "FP32", fps, avg_latency, observation)

        print(f"Results for {model_tag}: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_model_size_tests()
