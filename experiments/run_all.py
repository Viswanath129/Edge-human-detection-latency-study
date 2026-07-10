import subprocess
import os
import sys

def run_script(script_name):
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}")

    script_path = os.path.join("experiments", script_name)
    # Ensure experiments directory is in PYTHONPATH for the subprocess
    # This allows experiment scripts to import utils.py reliably
    env = os.environ.copy()
    experiments_dir = os.path.abspath("experiments")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = experiments_dir + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = experiments_dir

    # Capture the original working directory
    original_cwd = os.getcwd()
    try:
        # Run from root to maintain consistent relative paths if needed,
        # but the scripts are designed to find paths relative to themselves.
        result = subprocess.run([sys.executable, script_path], env=env)
        if result.returncode != 0:
            print(f"Error running {script_name}")
    finally:
        os.chdir(original_cwd)

def main():
    # Ensure directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py",
        "plot_results.py"
    ]

    for script in scripts:
        run_script(script)

    print("\nAll benchmarks and plotting completed.")

if __name__ == "__main__":
    main()
