import os
import sys
import subprocess

def main():
    print("Initializing Benchmark Suite Orchestrator...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, ".."))

    # Define absolute paths for directories and ensure they exist
    tables_dir = os.path.join(root_dir, "results", "tables")
    plots_dir = os.path.join(root_dir, "results", "plots")

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Ensured directories exist:\n - Tables: {tables_dir}\n - Plots: {plots_dir}")

    # Set up environmental variables for subprocesses
    env = os.environ.copy()
    env["FORCE_SYNTHETIC"] = "true"

    # Prepend/append the experiments directory to PYTHONPATH using os.pathsep
    # ensures cross-platform compatibility and local module resolution
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = current_dir + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = current_dir

    print(f"Configured PYTHONPATH: {env['PYTHONPATH']}")
    print(f"Configured FORCE_SYNTHETIC: {env['FORCE_SYNTHETIC']}")

    # List of scripts to execute sequentially
    scripts = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py",
        "plot_results.py"
    ]

    for script in scripts:
        script_path = os.path.join(current_dir, script)
        print("\n" + "="*60)
        print(f"Executing: {script}")
        print("="*60)

        # Invoke subprocess using sys.executable to maintain environment/interpreter consistency
        result = subprocess.run([sys.executable, script_path], env=env, cwd=root_dir)

        if result.returncode != 0:
            print(f"Error: {script} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

        print(f"Successfully finished execution of {script}")

    print("\n" + "="*60)
    print("All benchmark experiments completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
