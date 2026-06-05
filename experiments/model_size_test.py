from utils import benchmark_model, save_summary

def run_model_size_test():
    print("Starting Model Size Benchmark...")

    models = ["yolov8n.pt", "yolov8s.pt"]
    resolution = 640

    for m_path in models:
        m_name = m_path.split('.')[0]
        print(f"Testing model: {m_name}...")
        fps, latency = benchmark_model(m_path, imgsz=resolution)

        observation = "Lightweight for edge" if "yolov8n" in m_path else "Improved accuracy"
        save_summary(f"{resolution}x{resolution}", m_name, "FP32", fps, latency, observation)

        print(f"Results for {m_name}: {fps} FPS, {latency} ms")

if __name__ == "__main__":
    run_model_size_test()
