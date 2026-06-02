from utils import benchmark_model, save_summary

def run_precision_test():
    model_name = "yolov8n.pt"
    res = 640
    precisions = [
        {"half": False, "name": "FP32"},
        {"half": True, "name": "FP16"}
    ]

    for p in precisions:
        print(f"Benchmarking precision: {p['name']}")
        fps, latency = benchmark_model(model_name, imgsz=res, half=p['half'])

        observation = "Standard precision" if p['name'] == "FP32" else "Reduced precision for speedup (HW dependent)"
        save_summary(f"{res}x{res}", "yolov8n", p['name'], fps, latency, observation)

if __name__ == "__main__":
    run_precision_test()
