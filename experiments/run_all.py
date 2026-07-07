import subprocess
import os
import sys

def main():
    # Set current working directory to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Ensure result directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Add experiments to PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + os.path.join(project_root, "experiments")

    scripts = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py"
    ]

    for script in scripts:
        print(f"\n>>> Running {script}...")
        subprocess.run([sys.executable, script], check=True, env=env)

    print("\n>>> Generating plots...")
    # Change to plots directory to run the plotting script (it expects relative paths)
    plots_dir = os.path.join(project_root, "results", "plots")
    subprocess.run([sys.executable, "plot_results.py"], check=True, cwd=plots_dir)

    print("\nAll benchmarks and visualizations completed successfully.")

if __name__ == "__main__":
    main()
