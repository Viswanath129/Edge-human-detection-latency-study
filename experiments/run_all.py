import subprocess
import os
import sys

def run_script(script_name):
    print(f"\n--- Running {script_name} ---")
    script_path = os.path.join("experiments", script_name)
    # Ensure experiments directory is in PYTHONPATH for submodule imports
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + os.path.abspath("experiments")

    result = subprocess.run([sys.executable, script_path], env=env)
    if result.returncode != 0:
        print(f"Error: {script_name} failed with return code {result.returncode}")
        return False
    return True

def main():
    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py",
        "plot_results.py"
    ]

    for script in scripts:
        if not run_script(script):
            print("Orchestration stopped due to script failure.")
            sys.exit(1)

    print("\nAll experiments and plotting completed successfully.")

if __name__ == "__main__":
    main()
