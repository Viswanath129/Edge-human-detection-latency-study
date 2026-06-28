import os
import sys
import subprocess

def main():
    # Set the working directory to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    # Ensure directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py"
    ]

    python_exe = sys.executable

    for script in scripts:
        print(f"--- Running {script} ---")
        try:
            # Set PYTHONPATH to include experiments directory for utils import
            env = os.environ.copy()
            # Actually, the scripts themselves add their own directory to sys.path,
            # but setting PYTHONPATH is a good backup.
            current_pythonpath = env.get("PYTHONPATH", "")
            experiments_dir = os.path.join(project_root, "experiments")
            if current_pythonpath:
                env["PYTHONPATH"] = f"{current_pythonpath}{os.pathsep}{experiments_dir}"
            else:
                env["PYTHONPATH"] = experiments_dir

            subprocess.run([python_exe, script], check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

    print("--- Generating Plots ---")
    try:
        # Change to results/plots to run plot_results.py
        os.chdir(os.path.join(project_root, "results", "plots"))
        subprocess.run([python_exe, "plot_results.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()
