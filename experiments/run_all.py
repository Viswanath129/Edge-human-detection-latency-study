import subprocess
import sys
import os

def run_all():
    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Project root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    scripts = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py"
    ]

    # Add experiments to PYTHONPATH for subprocesses
    env = os.environ.copy()
    exp_dir = os.path.join(root_dir, "experiments")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{exp_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = exp_dir

    for script in scripts:
        print(f"=== Running {script} ===")
        try:
            subprocess.run([sys.executable, script], env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

    # Generate plots
    print("=== Generating Plots ===")
    plot_script = "results/plots/plot_results.py"
    try:
        # Plot script needs to be run from its directory or handle paths correctly
        plot_dir = os.path.dirname(os.path.abspath(plot_script))
        subprocess.run([sys.executable, "plot_results.py"], cwd=plot_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating plots: {e}")

    print("=== All experiments completed ===")

if __name__ == "__main__":
    run_all()
