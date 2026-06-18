import os
import sys

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import benchmark_model, save_summary

def main():
    # Comparing FP32 and FP16 at 640x640 for yolov8n
    model_path = "yolov8n.pt"
    res = 640

    for half in [False, True]:
        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=res, half=half)

        # Skip saving if FP16 was requested but not achieved (avoids overwriting FP32 results)
        if half and not actual_half:
            print("Skipping FP16 summary update: Hardware acceleration not available.")
            continue

        precision = "FP16" if actual_half else "FP32"
        res_str = f"{res}x{res}"
        obs = "Standard Precision" if not half else "Half Precision (Accelerated)"

        save_summary(res_str, "yolov8n", precision, fps, avg_latency, obs)

if __name__ == "__main__":
    main()
