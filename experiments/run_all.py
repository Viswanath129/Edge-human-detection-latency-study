import subprocess
import sys
import os

def main():
    # Ensure we are in the project root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root_dir)

    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py",
        "results/plots/plot_results.py"
    ]

    # Add experiments to PYTHONPATH for subprocesses
    env = os.environ.copy()
    experiments_dir = os.path.join(root_dir, "experiments")
    env["PYTHONPATH"] = experiments_dir + os.pathsep + env.get("PYTHONPATH", "")

    for script in scripts:
        print(f"\n{'='*50}")
        print(f"Running {script}...")
        print(f"{'='*50}")

        try:
            # Use sys.executable to ensure we use the same python environment
            subprocess.run([sys.executable, script], check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
            sys.exit(1)

    print("\nAll experiments and plotting completed successfully.")

if __name__ == "__main__":
    main()
