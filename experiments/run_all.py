import subprocess
import sys
import os

def main():
    # Set up environment
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    os.chdir(project_root)

    # Ensure directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py",
        "results/plots/plot_results.py"
    ]

    python_exe = sys.executable

    for script in scripts:
        print(f"\n{'='*50}")
        print(f"Running: {script}")
        print(f"{'='*50}")

        # Add experiments to PYTHONPATH for the subprocess
        env = os.environ.copy()
        experiments_path = os.path.join(project_root, "experiments")
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{experiments_path}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = experiments_path

        try:
            subprocess.run([python_exe, script], check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

    print("\nAll experiments and plotting completed.")

if __name__ == "__main__":
    main()
