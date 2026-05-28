import os
import sys
from utils import save_summary
from resolution_test import run_resolution_test
from precision_test import run_precision_test
from model_size_test import run_model_size_test

def main():
    print("Starting comprehensive benchmark suite...")

    all_results = []

    print("\n--- Running Resolution Tests ---")
    all_results.extend(run_resolution_test())

    print("\n--- Running Precision Tests ---")
    all_results.extend(run_precision_test())

    print("\n--- Running Model Size Tests ---")
    all_results.extend(run_model_size_test())

    print("\nSaving results to summary.csv...")
    save_summary(all_results)
    print("Done!")

if __name__ == "__main__":
    main()
