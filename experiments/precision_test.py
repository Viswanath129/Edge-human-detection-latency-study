from utils import benchmark_model, save_summary

def run_test():
    precisions = [False, True] # FP32, FP16
    model_path = "yolov8n.pt"
    res = 640

    for half in precisions:
        precision_name = "FP16" if half else "FP32"
        print(f"Testing precision: {precision_name}")
        avg_latency, fps, actual_half = benchmark_model(model_path, imgsz=res, half=half)

        actual_precision = "FP16" if actual_half else "FP32"
        obs = "Hardware acceleration enabled" if actual_half else "Standard precision"
        save_summary(f"{res}x{res}", "yolov8n", actual_precision, fps, avg_latency, obs)

if __name__ == "__main__":
    run_test()
