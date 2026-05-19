from utils import run_benchmark

if __name__ == "__main__":
    # Test YOLOv8 nano
    run_benchmark(
        model_path="yolov8n.pt",
        imgsz=640,
        experiment_name="model_nano"
    )

    # Test YOLOv8 small
    run_benchmark(
        model_path="yolov8s.pt",
        imgsz=640,
        experiment_name="model_small"
    )
