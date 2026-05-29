from utils import benchmark_model, save_summary

def main():
    precisions = [False, True] # FP32, FP16
    model_path = "yolov8n.pt"
    res = 640

    for half in precisions:
        prec_name = "FP16" if half else "FP32"
        print(f"Benchmarking precision: {prec_name}")
        fps, latency = benchmark_model(model_path, imgsz=res, half=half)

        observation = "Standard precision" if not half else "Half precision (Speedup varies by HW)"
        save_summary(f"{res}x{res}", "YOLOv8n", prec_name, fps, latency, observation)
        print(f"Results: {fps:.2f} FPS, {latency:.2f} ms")

if __name__ == "__main__":
    main()
