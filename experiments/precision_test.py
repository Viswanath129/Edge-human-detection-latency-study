from utils import benchmark_model, save_summary

def run_precision_test():
    model_name = "yolov8n.pt"
    res = 640
    precisions = [
        {"half": False, "label": "FP32"},
        {"half": True, "label": "FP16"}
    ]

    for p in precisions:
        print(f"\n--- Testing Precision: {p['label']} ---")
        fps, latency = benchmark_model(model_name, imgsz=res, half=p['half'])

        observation = "Standard precision" if p['label'] == "FP32" else "Half precision (may be slower on CPU)"
        save_summary(f"{res}x{res}", "yolov8n", p['label'], fps, latency, observation)

        print(f"Results for {p['label']}: {fps:.2f} FPS, {latency:.2f} ms")

if __name__ == "__main__":
    run_precision_test()
