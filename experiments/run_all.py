import subprocess
import sys
import os

def run_experiment(script_path):
    print(f"--- Running {script_path} ---")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + ":" + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, script_path], env=env)
    if result.returncode != 0:
        print(f"Error running {script_path}")

def main():
    # Ensure results directories exist
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs(os.path.join(root_dir, "results/tables"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "results/plots"), exist_ok=True)

    # Change to root dir for consistent paths
    os.chdir(root_dir)

    experiments = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py"
    ]

    for exp in experiments:
        run_experiment(exp)

    print("--- Generating Plots ---")
    run_experiment("results/plots/plot_results.py")

    print("Benchmark suite completed.")

if __name__ == "__main__":
    main()
