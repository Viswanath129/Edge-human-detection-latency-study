import subprocess
import os
import sys

def run_all():
    # Set base directory to project root
    script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_path)
    os.chdir(project_root)

    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Environment setup
    env = os.environ.copy()
    env['FORCE_SYNTHETIC'] = 'true'
    # Add experiments to PYTHONPATH so scripts can find utils.py
    env['PYTHONPATH'] = os.pathsep.join([env.get('PYTHONPATH', ''), script_path])

    scripts = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py"
    ]

    for script in scripts:
        print(f"\n--- Running {script} ---")
        subprocess.run([sys.executable, script], env=env, check=True)

    print("\n--- Generating Plots ---")
    plot_script = "plot_results.py"
    subprocess.run([sys.executable, plot_script], cwd="results/plots", env=env, check=True)

    print("\nAll experiments completed successfully.")

if __name__ == "__main__":
    run_all()
