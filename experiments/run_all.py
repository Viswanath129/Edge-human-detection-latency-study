import os
import sys
import subprocess

def main():
    print("==================================================")
    print("Starting Comprehensive YOLOv8 Benchmarking Suite")
    print("==================================================")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Automatically ensure that the results/tables and results/plots directories exist
    tables_dir = os.path.abspath(os.path.join(base_dir, "..", "results", "tables"))
    plots_dir = os.path.abspath(os.path.join(base_dir, "..", "results", "plots"))
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Set up PYTHONPATH environment variable for subprocesses
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{base_dir}{os.path.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = base_dir

    # Explicitly force synthetic frame source (headless / automated verification mode)
    env["FORCE_SYNTHETIC"] = "true"

    # List of test scripts and plotting script to execute sequentially
    scripts = [
        "resolution_test.py",
        "precision_test.py",
        "model_size_test.py",
        "plot_results.py"
    ]

    for script in scripts:
        script_path = os.path.join(base_dir, script)
        print(f"\n--- Running: {script} ---")

        # Spawn subprocess using sys.executable to maintain environment consistency
        # Maintain correct CWD (repository root or script directory - we will run in script directory)
        result = subprocess.run([sys.executable, script_path], env=env, capture_output=True, text=True, cwd=base_dir)

        if result.returncode != 0:
            print(f"Error occurred during execution of {script}:")
            print(result.stderr)
            sys.exit(result.returncode)
        else:
            print(result.stdout.strip())

    print("\n==================================================")
    print("Benchmarking Suite Executed and Plots Regenerated.")
    print("==================================================")

if __name__ == "__main__":
    main()
