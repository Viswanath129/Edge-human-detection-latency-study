import subprocess
import os
import sys

def run_script(script_path):
    print(f"\n--- Executing {script_path} ---")
    # Add current directory to PYTHONPATH so scripts can import utils
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get("PYTHONPATH", "")
    env["FORCE_SYNTHETIC"] = "true"  # Ensure headless compatibility

    result = subprocess.run([sys.executable, script_path], env=env)
    if result.returncode != 0:
        print(f"Error executing {script_path}")

def main():
    # Ensure results directories exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(base_dir, "results", "tables"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results", "plots"), exist_ok=True)

    experiments_dir = os.path.dirname(os.path.abspath(__file__))

    scripts = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py",
        "plot_results.py"
    ]

    for script in scripts:
        script_path = os.path.join(experiments_dir, script)
        if os.path.exists(script_path):
            run_script(script_path)
        else:
            print(f"Warning: {script} not found in {experiments_dir}")

    print("\nAll experiments completed. Results are in results/ directory.")

if __name__ == "__main__":
    main()
