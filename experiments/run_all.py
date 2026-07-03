import subprocess
import sys
import os

def run_script(script_path):
    print(f"\n{'='*40}")
    print(f"Running: {script_path}")
    print(f"{'='*40}")

    # Add experiments to PYTHONPATH
    env = os.environ.copy()
    experiments_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = experiments_dir + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run([sys.executable, script_path], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")

def main():
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Ensure directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "experiments/resolution_test.py",
        "experiments/precision_test.py",
        "experiments/model_size_test.py",
        "results/plots/plot_results.py"
    ]

    for script in scripts:
        run_script(script)

    print("\nAll benchmarks and visualizations completed successfully.")

if __name__ == "__main__":
    # Force synthetic if no display/camera (standard for these environments)
    if "FORCE_SYNTHETIC" not in os.environ:
        os.environ["FORCE_SYNTHETIC"] = "true"
    main()
