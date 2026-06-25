import os
import subprocess
import sys

def run_script(script_path):
    print(f"\n{'='*20}")
    print(f"Running {script_path}...")
    print(f"{'='*20}\n")

    # Use the same python interpreter
    result = subprocess.run([sys.executable, script_path], check=True)
    return result.returncode == 0

def main():
    # Ensure we are in the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Create necessary directories
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "experiments/resolution_test.py",
        "experiments/precision_test.py",
        "experiments/model_size_test.py"
    ]

    for script in scripts:
        if not run_script(script):
            print(f"Error running {script}")
            sys.exit(1)

    print("\nAll experiments completed. Generating plots...")

    # Run plotting script
    plot_script = "results/plots/plot_results.py"
    # The plot script might need to be run from its own directory or root
    # Based on its code: df = pd.read_csv('../tables/summary.csv')
    # So it should be run from results/plots
    plot_dir = os.path.join(project_root, "results", "plots")
    subprocess.run([sys.executable, "plot_results.py"], cwd=plot_dir, check=True)

    print("\nWorkflow finished successfully.")

if __name__ == "__main__":
    main()
