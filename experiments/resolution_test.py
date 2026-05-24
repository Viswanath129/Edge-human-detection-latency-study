from utils import run_benchmark

def main():
    model_path = "yolov8n.pt"

    # Test 640x640
    run_benchmark(model_path, resolution=640, observation="Standard resolution")

    # Test 416x416
    run_benchmark(model_path, resolution=416, observation="Reduced resolution for speed")

if __name__ == "__main__":
    main()
