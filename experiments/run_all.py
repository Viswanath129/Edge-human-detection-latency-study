import os
import sys
import subprocess

def run_all():
    # 1. Automatically ensure directories exist using absolute paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tables_dir = os.path.abspath(os.path.join(current_dir, "../results/tables"))
    plots_dir = os.path.abspath(os.path.join(current_dir, "../results/plots"))

    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 2. Setup the environment with FORCE_SYNTHETIC=true and custom PYTHONPATH
    env = os.environ.copy()
    env["FORCE_SYNTHETIC"] = "true"

    # Prepend experiments directory to PYTHONPATH using os.pathsep
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = f"{current_dir}{os.pathsep}{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = current_dir

    # 3. Define the benchmark scripts to execute sequentially
    scripts = [
        "resolution_test.py",
        "precision_test.py",
        "model_size_test.py",
        "plot_results.py"
    ]

    for script in scripts:
        script_path = os.path.join(current_dir, script)
        print(f"\n==================================================")
        print(f"Executing: {script}")
        print(f"==================================================")

        # 4. Use sys.executable to maintain Python environment consistency
        try:
            subprocess.run([sys.executable, script_path], env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script}: {e}")
            sys.exit(1)

    print("\n==================================================")
    print("All benchmarks and plotting executed successfully!")
    print("==================================================")

if __name__ == "__main__":
    run_all()
