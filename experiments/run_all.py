import os
import sys
import subprocess

def run_all_benchmarks():
    print("="*60)
    print("Starting Comprehensive Edge YOLO Benchmark Suite")
    print("="*60)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    # 1. Ensure results/tables and results/plots directories exist
    tables_dir = os.path.join(project_root, "results", "tables")
    plots_dir = os.path.join(project_root, "results", "plots")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Ensured directories exist:\n - {tables_dir}\n - {plots_dir}")

    # 2. Setup environment variables: FORCE_SYNTHETIC=true and prepend experiments to PYTHONPATH
    env = os.environ.copy()
    env["FORCE_SYNTHETIC"] = "true"

    # Prepend experiments directory to PYTHONPATH with cross-platform separator
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{current_dir}{os.path.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = current_dir

    print(f"Set environment FORCE_SYNTHETIC=true and PYTHONPATH={env['PYTHONPATH']}")

    # 3. Scripts to execute in order
    scripts = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py",
        "plot_results.py"
    ]

    for script in scripts:
        script_path = os.path.join(current_dir, script)
        print(f"\nExecuting: {script} ...")

        # Use sys.executable to maintain environment consistency
        # Maintain consistent CWD for the subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=current_dir,
            env=env,
            capture_output=False
        )

        if result.returncode != 0:
            print(f"Error: {script} failed with exit code {result.returncode}")
            sys.exit(result.returncode)

    print("\n" + "="*60)
    print("All benchmark experiments and plotting completed successfully!")
    print("="*60)

if __name__ == "__main__":
    run_all_benchmarks()
