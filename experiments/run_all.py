import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n{'='*20}")
    print(f"Running {script_path}...")
    print(f"{'='*20}")

    result = subprocess.run([sys.executable, script_path], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}")
    return result.returncode

def main():
    # Ensure directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "experiments/resolution_test.py",
        "experiments/precision_test.py",
        "experiments/model_size_test.py"
    ]

    for script in scripts:
        run_script(script)

    # Run plotting script
    print("\nGenerating plots...")
    plot_script = "results/plots/plot_results.py"
    # Need to run it from its own directory because it uses relative paths
    subprocess.run([sys.executable, "plot_results.py"], cwd="results/plots")

if __name__ == "__main__":
    main()
