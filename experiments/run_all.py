import subprocess
import os
import sys

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}")

    env = os.environ.copy()
    # Ensure experiments directory is in PYTHONPATH for module resolution
    experiments_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = f"{experiments_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["FORCE_SYNTHETIC"] = "true"

    result = subprocess.run([sys.executable, script_name], cwd=experiments_dir, env=env)
    if result.returncode != 0:
        print(f"Error running {script_name}")

def main():
    # Ensure result directories exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(base_dir, "results", "tables"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results", "plots"), exist_ok=True)

    scripts = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py",
        "plot_results.py"
    ]

    for script in scripts:
        run_script(script)

    print("\nBenchmark suite execution complete.")

if __name__ == "__main__":
    main()
