import os
from resolution_test import run_resolution_test
from precision_test import run_precision_test
from model_size_test import run_model_size_test

def main():
    print("Starting Comprehensive Benchmarking Suite...")

    # 1. Resolution Tests
    run_resolution_test()

    # 2. Precision Tests
    run_precision_test()

    # 3. Model Size Tests
    run_model_size_test()

    print("\nAll benchmarks completed successfully.")
    print("Results consolidated in results/tables/summary.csv")

if __name__ == "__main__":
    main()
