import sys
import os

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import run_benchmark, save_results

def main():
    model_path = "yolov8n.pt"
    res = 640
    precisions = [False, True]  # False = FP32, True = FP16

    for half in precisions:
        precision_name = "FP16" if half else "FP32"
        print(f"Running benchmark for Precision: {precision_name}")

        fps, avg_latency, latencies = run_benchmark(model_path, img_size=res, half=half)

        observation = "Standard precision" if not half else "Reduced precision, faster on supported hardware"
        save_results(res, "yolov8n", precision_name, fps, avg_latency, latencies, observation)

        print(f"Finished {precision_name}: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    main()
