import os
import sys
import subprocess

def run_all_benchmarks():
    print("Starting Comprehensive Benchmark Suite...")

    # Ensure that the results/tables and results/plots directories exist before execution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    os.makedirs(os.path.join(project_root, "results", "tables"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "results", "plots"), exist_ok=True)

    # Environment variables: FORCE_SYNTHETIC=true and append experiments to PYTHONPATH
    env = os.environ.copy()
    env["FORCE_SYNTHETIC"] = "true"

    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{current_dir}{os.path.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = current_dir

    scripts = [
        "resolution_test.py",
        "precision_test.py",
        "model_size_test.py"
    ]

    for script in scripts:
        script_path = os.path.join(current_dir, script)
        print(f"\nExecuting {script}...")

        # Invoke the same Python interpreter as the parent process
        result = subprocess.run([sys.executable, script_path], env=env, cwd=project_root)
        if result.returncode != 0:
            print(f"Error executing {script}. Return code: {result.returncode}")
            sys.exit(result.returncode)

    # Run plotting script
    plot_script = os.path.join(current_dir, "plot_results.py")
    print(f"\nExecuting {plot_script}...")
    result = subprocess.run([sys.executable, plot_script], env=env, cwd=project_root)
    if result.returncode != 0:
        print(f"Error executing plot_results.py. Return code: {result.returncode}")
        sys.exit(result.returncode)

    print("\nBenchmark Suite Completed Successfully.")

if __name__ == "__main__":
    run_all_benchmarks()
