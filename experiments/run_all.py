import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n{'='*20}")
    print(f"Running {script_path}...")
    print(f"{'='*20}\n")
    try:
        # Use sys.executable to maintain the same Python environment
        subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")

def main():
    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Note: FORCE_SYNTHETIC can be set via environment variable for headless environments

    scripts = [
        "experiments/resolution_test.py",
        "experiments/precision_test.py",
        "experiments/model_size_test.py"
    ]

    for script in scripts:
        run_script(script)

    print("\nAll benchmarks completed.")

if __name__ == "__main__":
    main()
