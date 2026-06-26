from ultralytics import YOLO
from utils import benchmark_model, save_summary

def run_model_size_test():
    models = ["yolov8n.pt", "yolov8s.pt"]

    for model_path in models:
        model_name = model_path.split(".")[0]
        print(f"Benchmarking model: {model_name}")
        model = YOLO(model_path)

        avg_latency, fps, _ = benchmark_model(model, imgsz=640)

        obs = "Lightweight nano model" if "nano" in model_name or "8n" in model_name else "Higher accuracy small model"
        save_summary("640x640", model_name, "FP32", fps, avg_latency, obs)

if __name__ == "__main__":
    run_model_size_test()
