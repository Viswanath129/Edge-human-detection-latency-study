import subprocess
import os

def run_experiment(script_name):
    print(f"\n{'='*40}")
    print(f"Running {script_name}...")
    print(f"{'='*40}\n")

    script_path = os.path.join("experiments", script_name)
    result = subprocess.run(["python3", script_path], capture_output=False, text=True)

    if result.returncode == 0:
        print(f"\n{script_name} completed successfully.")
    else:
        print(f"\n{script_name} failed with return code {result.returncode}.")

def main():
    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    experiments = [
        "resolution_test.py",
        "precision_test.py",
        "model_size_test.py"
    ]

    for exp in experiments:
        run_experiment(exp)

    print("\nAll experiments completed. Updating plots...")
    subprocess.run(["python3", "results/plots/plot_results.py"], capture_output=False, text=True)
    print("Done.")

if __name__ == "__main__":
    main()
