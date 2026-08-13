import os
import sys
import subprocess

def main():
    print("=== Initializing Orchestrated Benchmark Run ===")

    # Ensure correct output directories exist before executing the suite
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Individual test scripts
    scripts = [
        "experiments/resolution_test.py",
        "experiments/precision_test.py",
        "experiments/model_size_test.py"
    ]

    # Configure robust runtime environment variables to enable headless synthetic benchmarking
    env = os.environ.copy()
    env["FORCE_SYNTHETIC"] = "true"

    # Prepend experiments/ directory to PYTHONPATH using pathsep
    experiments_dir = os.path.join(os.getcwd(), "experiments")
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{experiments_dir}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = experiments_dir

    # Execute all tests using sys.executable to maintain environment consistency
    for script in scripts:
        print(f"\n>>> Executing experimental script: {script}")
        try:
            subprocess.run([sys.executable, script], env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during execution of {script}: {e}", file=sys.stderr)

    # Run the comparative plotting generation
    plot_script = "experiments/plot_results.py"
    print(f"\n>>> Generating comparative visualizations: {plot_script}")
    try:
        subprocess.run([sys.executable, plot_script], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating comparative plots: {e}", file=sys.stderr)

    print("\n=== All Benchmarking and Visualization Tasks Completed ===")

if __name__ == "__main__":
    main()
