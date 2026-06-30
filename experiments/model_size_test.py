import os
import sys

# Ensure utils can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_test():
    models = ["yolov8n.pt", "yolov8s.pt"]
    imgsz = 640

    for m_name in models:
        print(f"Benchmarking model: {m_name}")
        avg_latency, fps, _ = benchmark_model(m_name, imgsz=imgsz, half=False)

        obs = "Baseline Nano model" if "n" in m_name else "Improved accuracy, higher latency"
        save_summary(f"{imgsz}x{imgsz}", m_name, "FP32", fps, avg_latency, obs)

if __name__ == "__main__":
    run_test()
