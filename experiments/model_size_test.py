from utils import benchmark_model, save_summary

def main():
    models = ["yolov8n.pt", "yolov8s.pt"]
    res = 640

    for m in models:
        avg_latency, fps, _ = benchmark_model(m, imgsz=res)
        model_type = "yolov8n" if "n" in m else "yolov8s"
        observation = "Ultra-lightweight" if "n" in m else "Balanced accuracy/speed"
        save_summary(f"{res}x{res}", model_type, "FP32", fps, avg_latency, observation)

if __name__ == "__main__":
    main()
