import os
import subprocess
import sys

def ensure_dirs():
    """Ensure necessary directories exist."""
    dirs = ['results/tables', 'results/plots']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        # Touch .gitkeep if they don't exist to maintain structure
        gitkeep = os.path.join(d, '.gitkeep')
        if not os.path.exists(gitkeep):
            with open(gitkeep, 'w') as f:
                pass

def run_benchmarks():
    """Run all benchmark scripts."""
    scripts = [
        "experiments/resolution_test.py",
        "experiments/model_size_test.py",
        "experiments/precision_test.py"
    ]

    # Add current directory to PYTHONPATH so experiments can find utils.py
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}/experiments:{env.get('PYTHONPATH', '')}"

    for script in scripts:
        print(f"\n{'='*50}")
        print(f"Executing: {script}")
        print(f"{'='*50}")
        try:
            subprocess.run([sys.executable, script], env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script}: {e}")

def run_plotting():
    """Execute the plotting script."""
    print(f"\n{'='*50}")
    print("Generating Plots...")
    print(f"{'='*50}")
    try:
        # Run from results/plots directory to maintain internal path logic of plot_results.py
        cwd = os.path.join(os.getcwd(), 'results/plots')
        subprocess.run([sys.executable, 'plot_results.py'], cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    ensure_dirs()
    run_benchmarks()
    run_plotting()
    print("\nAll benchmarks and visualizations completed successfully.")
