import subprocess
import os
import sys

def main():
    # Set base directory to the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base_dir)

    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Force synthetic frames for headless environment
    os.environ["FORCE_SYNTHETIC"] = "true"

    experiments = [
        os.path.join("experiments", "resolution_test.py"),
        os.path.join("experiments", "model_size_test.py"),
        os.path.join("experiments", "precision_test.py")
    ]

    for exp in experiments:
        print(f"Running {exp}...")
        subprocess.run([sys.executable, exp], check=True)

    print("All experiments completed. Generating plots...")
    # Run plotting script
    plot_script = os.path.join("results", "plots", "plot_results.py")
    plot_cwd = os.path.join("results", "plots")
    subprocess.run([sys.executable, "plot_results.py"], cwd=plot_cwd, check=True)
    print("Orchestration complete.")

if __name__ == "__main__":
    main()
