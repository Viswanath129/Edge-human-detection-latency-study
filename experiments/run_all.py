import os
import sys
import subprocess

def main():
    # Ensure directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "experiments/resolution_test.py",
        "experiments/precision_test.py",
        "experiments/model_size_test.py"
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{os.path.join(os.getcwd(), 'experiments')}"

    for script in scripts:
        print(f"=== Executing {script} ===")
        try:
            # Use sys.executable to ensure we use the same python interpreter
            subprocess.run([sys.executable, script], env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

    print("=== All experiments completed. Generating plots... ===")
    try:
        # Fixed path for plot_results.py when using cwd
        subprocess.run([sys.executable, "plot_results.py"], cwd="results/plots", check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()
