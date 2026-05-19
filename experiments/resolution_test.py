from utils import run_benchmark

if __name__ == "__main__":
    # Test 640 resolution
    run_benchmark(
        model_path="yolov8n.pt",
        imgsz=640,
        experiment_name="resolution_640"
    )

    # Test 416 resolution
    run_benchmark(
        model_path="yolov8n.pt",
        imgsz=416,
        experiment_name="resolution_416"
    )
