import subprocess
import os
import sys

def run_experiment(script_name):
    print(f"\n{'='*20}")
    print(f"Running {script_name}...")
    print(f"{'='*20}")

    script_path = os.path.join("experiments", script_name)

    # Ensure experiments directory is in PYTHONPATH for the subprocess
    env = os.environ.copy()
    experiments_dir = os.path.abspath("experiments")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{experiments_dir}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = experiments_dir

    result = subprocess.run([sys.executable, script_path], env=env)
    if result.returncode != 0:
        print(f"Error running {script_name}")
    else:
        print(f"Successfully finished {script_name}")

def main():
    # Change to root directory if we are inside experiments/
    if os.path.basename(os.getcwd()) == "experiments":
        os.chdir("..")

    # Create results directories
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    experiments = [
        "resolution_test.py",
        "precision_test.py",
        "model_size_test.py"
    ]

    for exp in experiments:
        run_experiment(exp)

    # Finally, update plots
    print(f"\n{'='*20}")
    print("Updating plots...")
    print(f"{'='*20}")

    plot_script = os.path.join("results", "plots", "plot_results.py")
    # Change directory to where the plot script is to ensure relative paths work as expected by original script
    original_cwd = os.getcwd()
    os.chdir(os.path.join("results", "plots"))
    subprocess.run([sys.executable, "plot_results.py"])
    os.chdir(original_cwd)

    print("\nAll experiments and plotting completed successfully.")

if __name__ == "__main__":
    main()
