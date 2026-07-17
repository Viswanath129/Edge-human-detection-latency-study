import os
import sys
import subprocess

def main():
    print("==================================================")
    # 1. Resolve absolute paths for directories and scripts
    experiments_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(experiments_dir, ".."))

    tables_dir = os.path.join(project_root, "results", "tables")
    plots_dir = os.path.join(project_root, "results", "plots")

    # Ensure tables and plots directories exist
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Ensured results directories exist:\n - {tables_dir}\n - {plots_dir}")

    # 2. Setup the environment for subprocesses
    env = os.environ.copy()
    env["FORCE_SYNTHETIC"] = "true"  # Force synthetic numpy frames in headless environments

    # Prepend/append experiments directory to PYTHONPATH for module resolution
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{experiments_dir}{os.path.sep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = experiments_dir

    print(f"Configured PYTHONPATH: {env['PYTHONPATH']}")
    print(f"Set FORCE_SYNTHETIC: {env['FORCE_SYNTHETIC']}")
    print("==================================================")

    # List of scripts to execute sequentially
    scripts = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py",
        "plot_results.py"
    ]

    for script in scripts:
        script_path = os.path.join(experiments_dir, script)
        print(f"\n>>> Executing script: {script} ...")

        # Invoke python interpreter with sys.executable to maintain environment consistency
        try:
            subprocess.run(
                [sys.executable, script_path],
                env=env,
                check=True,
                text=True
            )
            print(f">>> {script} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error executing {script}: {e}")
            sys.exit(1)

    print("\n==================================================")
    print("Full benchmark suite orchestrated and completed successfully!")
    print("==================================================")

if __name__ == "__main__":
    main()
