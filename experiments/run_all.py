import subprocess
import sys
import os

def run_script(script_name):
    print(f"--- Running {script_name} ---")
    result = subprocess.run([sys.executable, script_name], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}")
        return False
    return True

def main():
    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "experiments/resolution_test.py",
        "experiments/precision_test.py",
        "experiments/model_size_test.py"
    ]

    for script in scripts:
        if not run_script(script):
            sys.exit(1)

    print("--- Generating Plots ---")
    plot_script = "results/plots/plot_results.py"
    # Run plot script from its own directory for correct relative paths
    plot_dir = os.path.dirname(plot_script)
    result = subprocess.run([sys.executable, os.path.basename(plot_script)], cwd=plot_dir)

    if result.returncode == 0:
        print("All experiments completed successfully.")
    else:
        print("Failed to generate plots.")
        sys.exit(1)

if __name__ == "__main__":
    main()
