from utils import run_benchmark

def main():
    # Benchmark FP32
    run_benchmark(
        model_name="yolov8n.pt",
        resolution=640,
        half=False,
        observation="Standard precision"
    )

    # Benchmark FP16
    run_benchmark(
        model_name="yolov8n.pt",
        resolution=640,
        half=True,
        observation="Half precision (hardware dependent)"
    )

if __name__ == "__main__":
    main()
