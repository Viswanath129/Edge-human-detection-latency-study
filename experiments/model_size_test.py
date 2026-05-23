from utils import run_benchmark

def main():
    # Benchmark YOLOv8n (nano)
    run_benchmark(
        model_name="yolov8n.pt",
        resolution=640,
        half=False,
        observation="Nano model (fastest)"
    )

    # Benchmark YOLOv8s (small)
    run_benchmark(
        model_name="yolov8s.pt",
        resolution=640,
        half=False,
        observation="Small model (balanced)"
    )

if __name__ == "__main__":
    main()
