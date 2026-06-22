import os
import sys
import subprocess

def main():
    # Set project root and PYTHONPATH
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    sys.path.append(os.path.join(project_root, "experiments"))

    # Ensure directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    print("--- Starting Benchmarking Suite ---")

    # Scripts to run
    scripts = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py"
    ]

    python_exe = sys.executable

    for script in scripts:
        print(f"\nRunning {script}...")
        try:
            # We use FORCE_SYNTHETIC=true to ensure it works in headless/CI-like environments
            env = os.environ.copy()
            env["FORCE_SYNTHETIC"] = "true"
            env["PYTHONPATH"] = f"{project_root}:{project_root}/experiments"

            subprocess.run([python_exe, script], check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

    print("\n--- Generating Plots ---")
    try:
        plot_script = "results/plots/plot_results.py"
        # The original plot_results.py uses relative paths assuming it is run from its directory
        # or it uses '../tables/summary.csv'
        # Let's run it from its directory to be safe if it's not yet refactored.
        plot_dir = os.path.join(project_root, "results", "plots")
        subprocess.run([python_exe, "plot_results.py"], check=True, cwd=plot_dir)
    except subprocess.CalledProcessError as e:
        print(f"Error generating plots: {e}")

    print("\n--- Benchmark Suite Complete ---")

if __name__ == "__main__":
    main()
