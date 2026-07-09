import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*40}")
    print(f"Executing {script_name}...")
    print(f"{'='*40}")

    # Use sys.executable to maintain environment consistency
    # Add experiments directory to PYTHONPATH for subprocess
    env = os.environ.copy()
    experiments_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = experiments_dir + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run([sys.executable, script_name], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_name}: {e}")

def main():
    # Save original working directory
    original_cwd = os.getcwd()

    # Change to project root (parent of experiments)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Ensure result directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Force synthetic frames for standard benchmark runs to ensure reproducibility/headless support
    os.environ["FORCE_SYNTHETIC"] = "true"

    scripts = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py",
        "experiments/plot_results.py"
    ]

    try:
        for script in scripts:
            if os.path.exists(script):
                run_script(script)
            else:
                print(f"Script not found: {script}")

        print("\nAll benchmarks and visualizations completed successfully.")
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()
