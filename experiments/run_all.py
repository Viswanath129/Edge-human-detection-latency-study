import subprocess
import os
import sys

def run_experiment(script_name):
    print(f"\n{'='*20}")
    print(f"Running {script_name}...")
    print(f"{'='*20}\n")

    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)

    # Use sys.executable to ensure we use the same python interpreter
    result = subprocess.run([sys.executable, script_path], capture_output=False)

    if result.returncode != 0:
        print(f"Error running {script_name}")
    else:
        print(f"Successfully completed {script_name}")

def main():
    # Ensure results directories exist
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(root_dir, "results/tables"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "results/plots"), exist_ok=True)

    # List of experiments to run
    experiments = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py"
    ]

    for exp in experiments:
        run_experiment(exp)

    # Run plotting script
    print(f"\n{'='*20}")
    print("Generating Plots...")
    print(f"{'='*20}\n")

    plot_script = os.path.join(root_dir, "results/plots/plot_results.py")
    subprocess.run([sys.executable, plot_script], cwd=os.path.join(root_dir, "results/plots"))

    print("\nAll experiments and visualizations completed.")

if __name__ == "__main__":
    main()
