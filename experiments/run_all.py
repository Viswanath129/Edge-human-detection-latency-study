import subprocess
import sys
import os

def main():
    # Set working directory to project root
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    os.chdir(project_root)

    # Ensure output directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    experiments = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py"
    ]

    for exp in experiments:
        print(f"\n{'='*50}")
        print(f"Running experiment: {exp}")
        print(f"{'='*50}")

        result = subprocess.run([sys.executable, exp], capture_output=False)

        if result.returncode != 0:
            print(f"Experiment {exp} failed with return code {result.returncode}")
        else:
            print(f"Experiment {exp} completed successfully")

    # Run plotting script
    print(f"\n{'='*50}")
    print("Generating plots...")
    print(f"{'='*50}")

    # Change to plots directory to run plot script correctly if it uses relative paths
    os.chdir(os.path.join(project_root, "results", "plots"))
    subprocess.run([sys.executable, "plot_results.py"])

    print("\nAll experiments and plotting completed.")

if __name__ == "__main__":
    main()
