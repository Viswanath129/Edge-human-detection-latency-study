import os
import sys

# Ensure the experiments directory is in the path
sys.path.append(os.path.dirname(__file__))

from resolution_test import run_resolution_test
from precision_test import run_precision_test
from model_size_test import run_model_size_test

def main():
    print("Starting Comprehensive Benchmark Suite...")

    # Set environment to use synthetic frames if no webcam is present
    os.environ["FORCE_SYNTHETIC"] = "true"

    run_resolution_test()
    run_precision_test()
    run_model_size_test()

    print("\nAll benchmarks completed successfully.")
    print("Summary results available in results/tables/summary.csv")

if __name__ == "__main__":
    main()
