from ultralytics import YOLO
from utils import benchmark_model, save_summary

def run_model_size_test():
    resolutions = [640]
    model_variants = ["yolov8n", "yolov8s"]

    for model_name in model_variants:
        print(f"Benchmarking {model_name}...")
        model = YOLO(f"{model_name}.pt")

        for res in resolutions:
            avg_latency, fps, _ = benchmark_model(model, imgsz=res)
            res_str = f"{res}x{res}"
            obs = "Nano model (Fastest)" if "n" in model_name else "Small model (Balanced)"
            save_summary(res_str, model_name, "FP32", fps, avg_latency, obs)
            print(f"{model_name} @ {res_str}: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_model_size_test()
