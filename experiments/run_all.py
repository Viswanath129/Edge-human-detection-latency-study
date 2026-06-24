import os
import sys
import subprocess

def run_experiment(script_name):
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}")

    # Ensure experiments directory is in PYTHONPATH
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = f"{os.path.join(project_root, 'experiments')}:{env.get('PYTHONPATH', '')}"
    env["FORCE_SYNTHETIC"] = "true"

    result = subprocess.run([sys.executable, f"experiments/{script_name}"], env=env, cwd=project_root)
    if result.returncode != 0:
        print(f"Error running {script_name}")

def main():
    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    experiments = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py"
    ]

    for exp in experiments:
        run_experiment(exp)

    print(f"\n{'='*50}")
    print("Generating updated plots...")
    print(f"{'='*50}")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "plot_results.py"], cwd=os.path.join(project_root, "results/plots"))

    print("\nAll experiments completed. Results are available in results/ and documented in report/research_note.md")

if __name__ == "__main__":
    main()
