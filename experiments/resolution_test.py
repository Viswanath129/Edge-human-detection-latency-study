import sys
import os

# Add the current directory to sys.path to allow importing utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_resolution_tests():
    resolutions = [640, 416]
    model_path = "yolov8n.pt"

    print(f"Starting resolution tests with {model_path}...")

    for res in resolutions:
        print(f"Testing resolution: {res}x{res}")
        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=res)

        precision = "FP16" if actual_half else "FP32"
        observation = "Standard resolution" if res == 640 else "Lower resolution for speed"

        save_summary(
            resolution=f"{res}x{res}",
            model_name="YOLOv8n",
            precision=precision,
            fps=fps,
            latency=avg_latency,
            observation=observation
        )
        print(f"Resolution {res}x{res} complete: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_resolution_tests()
