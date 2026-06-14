import subprocess
import sys
import os

def run_experiment(script_path):
    print(f"\n{'='*40}")
    print(f"Running: {script_path}")
    print(f"{'='*40}")

    # Use the same python interpreter as the current process
    # Set FORCE_SYNTHETIC=true for automated/headless runs
    env = os.environ.copy()
    env["FORCE_SYNTHETIC"] = "true"

    result = subprocess.run([sys.executable, script_path], env=env)
    if result.returncode != 0:
        print(f"Error running {script_path}")
    else:
        print(f"Finished: {script_path}")

def main():
    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    experiments = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py"
    ]

    for exp in experiments:
        run_experiment(exp)

    print("\nAll experiments completed.")

    # Run plotting script
    print("\nGenerating plots...")
    plot_script = "results/plots/plot_results.py"
    if os.path.exists(plot_script):
        # The plotting script expects to be run from results/plots to find ../tables/summary.csv
        subprocess.run([sys.executable, "plot_results.py"], cwd="results/plots")
    else:
        print(f"Plotting script not found: {plot_script}")

if __name__ == "__main__":
    main()
