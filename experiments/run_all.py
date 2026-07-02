import subprocess
import sys
import os

def run_all():
    # Set project root relative to this script
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "experiments/resolution_test.py",
        "experiments/precision_test.py",
        "experiments/model_size_test.py"
    ]

    # Add experiments to PYTHONPATH for local imports in subprocesses
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(project_root, "experiments") + os.pathsep + env.get("PYTHONPATH", "")

    for script_rel_path in scripts:
        script_abs_path = os.path.join(project_root, script_rel_path)
        print(f"\n{'='*20}")
        print(f"Running: {script_rel_path}")
        print(f"{'='*20}")

        try:
            subprocess.run([sys.executable, script_abs_path], check=True, env=env, cwd=project_root)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

    # Run plotting script
    print(f"\n{'='*20}")
    print("Generating Plots")
    print(f"{'='*20}")
    subprocess.run([sys.executable, "results/plots/plot_results.py"], env=env, cwd=project_root)

if __name__ == "__main__":
    run_all()
