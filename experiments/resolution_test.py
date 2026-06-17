from ultralytics import YOLO
from utils import benchmark_model, save_summary

def run_resolution_test():
    model_name = "yolov8n"
    model = YOLO(f"{model_name}.pt")
    resolutions = [640, 416]

    for res in resolutions:
        print(f"Benchmarking {model_name} at {res}x{res} resolution...")
        avg_latency, fps, _ = benchmark_model(model, imgsz=res)

        res_str = f"{res}x{res}"
        obs = "Higher detection quality" if res == 640 else "Faster Inference"
        save_summary(res_str, model_name, "FP32", fps, avg_latency, obs)

        print(f"{res_str}: {fps:.2f} FPS, {avg_latency:.2f} ms")

if __name__ == "__main__":
    run_resolution_test()
