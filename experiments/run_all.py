import subprocess
import os
import sys

def run_script(script_path):
    print(f"--- Running {script_path} ---")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}:")
        print(result.stderr)
    else:
        print(result.stdout)

def main():
    # Ensure result directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py"
    ]

    for script in scripts:
        run_script(script)

    # Run plotting script
    print("--- Generating Plots ---")
    plot_script = "plot_results.py"
    result = subprocess.run([sys.executable, plot_script], cwd="results/plots", capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {plot_script}:")
        print(result.stderr)
    else:
        print(result.stdout)

    print("All benchmarks and plots completed.")

if __name__ == "__main__":
    main()
