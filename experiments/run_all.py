import os
import subprocess
import sys

def run_all_experiments():
    # Set PYTHONPATH to include the experiments directory for local imports
    env = os.environ.copy()
    experiments_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = experiments_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["FORCE_SYNTHETIC"] = "true"  # Ensure it runs in headless environments

    scripts = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py",
        "plot_results.py"
    ]

    # Ensure results directories exist
    root_dir = os.path.dirname(experiments_dir)
    os.makedirs(os.path.join(root_dir, "results/tables"), exist_ok=True)
    os.makedirs(os.path.join(root_dir, "results/plots"), exist_ok=True)

    for script in scripts:
        print(f"\n{'='*20}")
        print(f"Running {script}...")
        print(f"{'='*20}")

        script_path = os.path.join(experiments_dir, script)
        result = subprocess.run([sys.executable, script_path], env=env, cwd=experiments_dir)

        if result.returncode != 0:
            print(f"Error: {script} failed with return code {result.returncode}")
            # We continue to other scripts even if one fails
        else:
            print(f"Successfully completed {script}")

if __name__ == "__main__":
    run_all_experiments()
