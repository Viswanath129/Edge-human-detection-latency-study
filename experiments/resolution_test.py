from utils import run_benchmark

def main():
    # Benchmark 640x640
    run_benchmark(
        model_name="yolov8n.pt",
        resolution=640,
        half=False,
        observation="Base resolution"
    )

    # Benchmark 416x416
    run_benchmark(
        model_name="yolov8n.pt",
        resolution=416,
        half=False,
        observation="Lower resolution for speed"
    )

if __name__ == "__main__":
    main()
