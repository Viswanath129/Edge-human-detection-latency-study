import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import benchmark_model, save_summary

def run_precision_test():
    res = 640
    model_name = "yolov8n.pt"
    precisions = [False, True] # False=FP32, True=FP16

    for half in precisions:
        prec_label = "FP16" if half else "FP32"
        print(f"\n--- Testing Precision: {prec_label} ---")

        avg_latency, fps, actual_half = benchmark_model(model_name=model_name, imgsz=res, half=half)

        if half and not actual_half:
            print("Skipping FP16 result save due to lack of hardware support.")
            continue

        save_summary(
            resolution=res,
            model_name=model_name.split('.')[0],
            precision=prec_label,
            fps=fps,
            latency=avg_latency,
            observation="Half precision impact" if half else "Baseline precision"
        )

if __name__ == "__main__":
    run_precision_test()
