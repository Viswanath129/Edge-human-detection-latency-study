from utils import run_benchmark

def main():
    # Compare FP32 vs FP16 precision
    model_path = "yolov8n.pt"

    # FP32 (default)
    run_benchmark(model_path, resolution=640, half=False, observation="Full precision (FP32)")

    # FP16 (half)
    # Note: FP16 is often slower on CPU. This script is intended for hardware with FP16 support (e.g., GPU/NPU).
    run_benchmark(model_path, resolution=640, half=True, observation="Half precision (FP16)")

if __name__ == "__main__":
    main()
