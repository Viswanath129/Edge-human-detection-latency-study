import sys
import os

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import run_benchmark, save_results

def main():
    models = {
        "yolov8n": "yolov8n.pt",
        "yolov8s": "yolov8s.pt"
    }
    res = 640

    for name, path in models.items():
        print(f"Running benchmark for Model: {name}")

        fps, avg_latency, latencies = run_benchmark(path, img_size=res)

        observation = "Lightweight (nano)" if name == "yolov8n" else "Medium (small)"
        save_results(res, name, "FP32", fps, avg_latency, latencies, observation)

        print(f"Finished {name}: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    main()
