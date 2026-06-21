import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_model_size_test():
    res = 640
    models = ["yolov8n.pt", "yolov8s.pt"]

    for m_name in models:
        print(f"\n--- Testing Model: {m_name} ---")
        avg_latency, fps, _ = benchmark_model(model_name=m_name, imgsz=res, half=False)

        m_short = m_name.split('.')[0]
        obs = "Ultra-lightweight" if "n" in m_short else "Small variant"

        save_summary(
            resolution=res,
            model_name=m_short,
            precision="FP32",
            fps=fps,
            latency=avg_latency,
            observation=obs
        )

if __name__ == "__main__":
    run_model_size_test()
