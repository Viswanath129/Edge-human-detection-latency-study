import subprocess
import sys
import os

def run_script(script_path):
    print(f"Executing {script_path}...")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Errors in {script_path}:\n{result.stderr}")

def main():
    # Ensure result directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "experiments/resolution_test.py",
        "experiments/precision_test.py",
        "experiments/model_size_test.py"
    ]

    for script in scripts:
        run_script(script)

    print("--- All benchmarks complete. Generating plots... ---")
    plotting_script = "results/plots/plot_results.py"
    if os.path.exists(plotting_script):
        # Run plotting script from its local directory to handle relative paths
        subprocess.run([sys.executable, "plot_results.py"], cwd="results/plots")
        print("Plots updated in results/plots/")
    else:
        print(f"Warning: Plotting script {plotting_script} not found.")

if __name__ == "__main__":
    main()
