import sys
import os

# Add the current directory to sys.path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def run_model_size_test():
    resolution = 640
    models = ["yolov8n.pt", "yolov8s.pt"]

    for m_name in models:
        avg_latency, fps, _ = benchmark_model(m_name, imgsz=resolution)

        observation = "Baseline nano model" if "n" in m_name else "Small model (more params)"
        save_summary(resolution, m_name.replace(".pt", ""), "FP32", fps, avg_latency, observation)

        print(f"Model: {m_name}, FPS: {fps:.2f}, Latency: {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_model_size_test()
