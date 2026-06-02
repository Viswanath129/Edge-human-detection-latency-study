import os
import sys

# Ensure experiments directory is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from resolution_test import run_resolution_test
from precision_test import run_precision_test
from model_size_test import run_model_size_test

def run_all_experiments():
    print("=== Starting Comprehensive Benchmark Suite ===")

    print("\n[1/3] Running Resolution Tests...")
    run_resolution_test()

    print("\n[2/3] Running Precision Tests...")
    run_precision_test()

    print("\n[3/3] Running Model Size Tests...")
    run_model_size_test()

    print("\n=== Benchmarking Complete ===")

if __name__ == "__main__":
    run_all_experiments()
