import subprocess
import os
import sys

def run_script(script_name):
    print(f"\n{'='*20}")
    print(f"Executing {script_name}...")
    print(f"{'='*20}\n")

    # Ensure experiments directory is in path for imports
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.path.dirname(os.path.abspath(__file__))}{os.pathsep}{env.get('PYTHONPATH', '')}"

    # Force synthetic frames for headless/automated run
    env["FORCE_SYNTHETIC"] = "true"

    try:
        subprocess.run([sys.executable, os.path.join("experiments", script_name)], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

def main():
    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Define experiments to run
    experiments = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py"
    ]

    for exp in experiments:
        run_script(exp)

    # Generate plots
    print(f"\n{'='*20}")
    print("Generating Plots...")
    print(f"{'='*20}\n")

    try:
        # Change dir to results/plots to run the plotting script
        original_cwd = os.getcwd()
        os.chdir("results/plots")
        subprocess.run([sys.executable, "plot_results.py"], check=True)
        os.chdir(original_cwd)
    except Exception as e:
        print(f"Error generating plots: {e}")
        os.chdir(original_cwd)

    print("\nBenchmark Suite Completed.")

if __name__ == "__main__":
    main()
