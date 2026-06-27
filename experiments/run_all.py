import subprocess
import sys
import os

def run_experiment(script_name):
    print(f"\n{'='*40}")
    print(f"Running {script_name}...")
    print(f"{'='*40}")

    script_path = os.path.join("experiments", script_name)
    # Ensure we use the same python interpreter
    result = subprocess.run([sys.executable, script_path], capture_output=False)

    if result.returncode != 0:
        print(f"Error running {script_name}")
    else:
        print(f"Successfully completed {script_name}")

def main():
    # Set CWD to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Ensure directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Optional: Force synthetic frames if not on a device with webcam
    # os.environ["FORCE_SYNTHETIC"] = "true"

    experiments = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py"
    ]

    for exp in experiments:
        run_experiment(exp)

    print("\nAll experiments completed. Generating plots...")
    subprocess.run([sys.executable, "results/plots/plot_results.py"])

if __name__ == "__main__":
    main()
