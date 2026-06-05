import os
import subprocess

def run_benchmarks():
    print("=== Starting Full Benchmark Suite ===")

    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # Set up environment
    env = os.environ.copy()
    env["FORCE_SYNTHETIC"] = "true"
    # Ensure experiments directory is in PYTHONPATH for utils import
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{current_dir}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = current_dir

    scripts = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py"
    ]

    for script in scripts:
        script_path = os.path.join(current_dir, script)
        print(f"\n--- Running {script} ---")
        try:
            subprocess.run(["python3", script_path], env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

    print("\n=== All Benchmarks Completed ===")

    # Run plotting script
    print("\n--- Generating Plots ---")
    plot_script = os.path.join(current_dir, "../results/plots/plot_results.py")
    try:
        subprocess.run(["python3", os.path.abspath(plot_script)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    run_benchmarks()
