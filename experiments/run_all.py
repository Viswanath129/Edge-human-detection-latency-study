import subprocess
import os
import sys

def run_script(script_name):
    print(f"\n{'='*20}")
    print(f"Executing: {script_name}")
    print(f"{'='*20}")

    # Ensure we use the correct python interpreter and environment
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get("PYTHONPATH", "")
    env["FORCE_SYNTHETIC"] = "true" # Ensure it runs in headless environments

    result = subprocess.run([sys.executable, script_name], env=env, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Error executing {script_name}")
    return result.returncode

def main():
    # Change directory to the root of the project
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    scripts = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py"
    ]

    for script in scripts:
        run_script(script)

    print("\nAll experiments completed.")

    # Run plotting script
    print("\nGenerating plots...")
    run_script("results/plots/plot_results.py")

if __name__ == "__main__":
    main()
