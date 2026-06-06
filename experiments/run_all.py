import os
from resolution_test import run_resolution_test
from precision_test import run_precision_test
from model_size_test import run_model_size_test

def run_all_benchmarks():
    print("Starting Comprehensive Benchmarking Suite...")

    run_resolution_test()
    run_precision_test()
    run_model_size_test()

    print("\nAll benchmarks completed. Results consolidated in results/tables/summary.csv")

if __name__ == "__main__":
    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    run_all_benchmarks()
