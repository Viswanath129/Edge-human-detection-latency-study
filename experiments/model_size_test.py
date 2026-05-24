from utils import run_benchmark

def main():
    # Compare yolov8n (nano) vs yolov8s (small)

    # yolov8n.pt at 640x640
    run_benchmark("yolov8n.pt", resolution=640, observation="Nano model (baseline)")

    # yolov8s.pt at 640x640
    run_benchmark("yolov8s.pt", resolution=640, observation="Small model (higher capacity)")

if __name__ == "__main__":
    main()
