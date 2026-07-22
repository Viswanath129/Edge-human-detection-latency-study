import os
import sys
import subprocess

def main():
    print("==================================================")
    print("YOLO EDGE BENCHMARK SUITE ORCHESTRATOR")
    print("==================================================")

    # 1. Determine absolute paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 2. Ensure output directories exist
    tables_dir = os.path.join(project_root, 'results', 'tables')
    plots_dir = os.path.join(project_root, 'results', 'plots')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Verified directories:\n  Tables: {tables_dir}\n  Plots: {plots_dir}")

    # 3. Setup environment for child processes
    # Setup PYTHONPATH with cross-platform separator and set FORCE_SYNTHETIC=true
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{script_dir}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = script_dir

    env["FORCE_SYNTHETIC"] = "true"
    print("Environment configured: FORCE_SYNTHETIC=true, PYTHONPATH added.")

    # 4. List of individual benchmark scripts to run
    benchmark_scripts = [
        os.path.join(script_dir, "resolution_test.py"),
        os.path.join(script_dir, "model_size_test.py"),
        os.path.join(script_dir, "precision_test.py"),
    ]

    # 5. Execute benchmark scripts using sys.executable (same interpreter)
    for script in benchmark_scripts:
        script_name = os.path.basename(script)
        print("\n--------------------------------------------------")
        print(f"Running benchmark: {script_name}")
        print("--------------------------------------------------")

        try:
            # Use sys.executable to ensure the same Python environment is used
            res = subprocess.run([sys.executable, script], env=env, check=True, capture_output=False)
            print(f"Successfully completed: {script_name}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {script_name}: {e}")
            sys.exit(1)

    # 6. Run plotting script
    plotting_script = os.path.join(script_dir, "plot_results.py")
    print("\n--------------------------------------------------")
    print("Generating Comparative Visualization Plots")
    print("--------------------------------------------------")
    try:
        subprocess.run([sys.executable, plotting_script], env=env, check=True, capture_output=False)
        print("Successfully generated all plots.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while generating plots: {e}")
        sys.exit(1)

    print("\n==================================================")
    print("BENCHMARK ORCHESTRATION COMPLETED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
