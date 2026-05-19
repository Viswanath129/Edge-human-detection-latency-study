from utils import run_benchmark

if __name__ == "__main__":
    # Test FP32 precision
    run_benchmark(
        model_path="yolov8n.pt",
        imgsz=640,
        half=False,
        experiment_name="precision_fp32"
    )

    # Test FP16 precision
    run_benchmark(
        model_path="yolov8n.pt",
        imgsz=640,
        half=True,
        experiment_name="precision_fp16"
    )
