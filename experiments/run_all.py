import os
import sys
import subprocess

def run_all():
    print("Starting comprehensive benchmark suite...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, '..'))

    # Ensure output directories exist before executing the benchmark suite
    os.makedirs(os.path.join(repo_root, 'results/tables'), exist_ok=True)
    os.makedirs(os.path.join(repo_root, 'results/plots'), exist_ok=True)

    # Build the environment dictionary
    env = os.environ.copy()

    # Prepend experiments directory to PYTHONPATH with cross-platform compatibility (os.pathsep)
    experiments_dir = os.path.join(repo_root, 'experiments')
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = experiments_dir + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = experiments_dir

    # Explicitly set FORCE_SYNTHETIC=true to ensure headless/reliable execution
    env['FORCE_SYNTHETIC'] = 'true'

    # List of scripts to run in order
    scripts = [
        'resolution_test.py',
        'model_size_test.py',
        'precision_test.py',
        'plot_results.py'
    ]

    for script in scripts:
        script_path = os.path.join(experiments_dir, script)
        print(f"\n--- Executing {script} ---")
        try:
            # Run using sys.executable to maintain environment consistency
            subprocess.run(
                [sys.executable, script_path],
                env=env,
                cwd=experiments_dir, # Ensure correct CWD for absolute path resolving in sub-scripts
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script}: {e}")
            sys.exit(1)

    print("\nAll benchmark tests executed and plots regenerated successfully.")

if __name__ == '__main__':
    run_all()
