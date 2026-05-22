import sys
import os

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import run_benchmark, save_results

def main():
    model_path = "yolov8n.pt"
    resolutions = [640, 416]

    for res in resolutions:
        print(f"Running benchmark for Resolution: {res}x{res}")
        fps, avg_latency, latencies = run_benchmark(model_path, img_size=res)

        observation = "Higher detection quality" if res == 640 else "Faster Inference"
        save_results(res, "yolov8n", "FP32", fps, avg_latency, latencies, observation)

        print(f"Finished {res}x{res}: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    main()
