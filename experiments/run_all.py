import subprocess
import os
import sys

def main():
    # Ensure result directories exist
    os.makedirs('results/tables', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)

    scripts = [
        'experiments/resolution_test.py',
        'experiments/model_size_test.py',
        'experiments/precision_test.py'
    ]

    for script in scripts:
        print(f"--- Running {script} ---")
        try:
            # Use sys.executable to ensure we use the same environment
            # Add experiments dir to PYTHONPATH so scripts can import utils
            env = os.environ.copy()
            env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get('PYTHONPATH', '')

            subprocess.run([sys.executable, script], check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

    # Run plotting script
    print("--- Generating Plots ---")
    try:
        # Plotting script might expect to be run from results/plots
        subprocess.run([sys.executable, 'plot_results.py'], cwd='results/plots', check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating plots: {e}")

if __name__ == "__main__":
    main()
