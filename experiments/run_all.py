import subprocess
import os
import sys

def main():
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    original_cwd = os.getcwd()

    os.chdir(project_root)

    try:
        # Ensure results directories exist
        os.makedirs("results/tables", exist_ok=True)
        os.makedirs("results/plots", exist_ok=True)

        print("=== Starting Comprehensive Benchmark Suite ===\n")

        # Add experiments to PYTHONPATH for subprocesses
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([script_dir, env.get("PYTHONPATH", "")])

        scripts = [
            "experiments/resolution_test.py",
            "experiments/model_size_test.py",
            "experiments/precision_test.py"
        ]

        for script in scripts:
            print(f"\n>> Executing {script}...")
            try:
                subprocess.run([sys.executable, script], env=env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing {script}: {e}")

        print("\n>> Generating Plots...")
        try:
            # Run plot script from its directory to satisfy relative path expectations
            plot_script_dir = os.path.join(project_root, "results", "plots")
            subprocess.run([sys.executable, "plot_results.py"], env=env, check=True, cwd=plot_script_dir)
        except subprocess.CalledProcessError as e:
            print(f"Error generating plots: {e}")

        print("\n=== All Benchmarks Completed Successfully ===")
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()
