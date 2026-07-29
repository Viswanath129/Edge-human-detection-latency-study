import os
import sys
import subprocess

def main():
    print("Initializing Benchmark Suite Orchestrator...")

    # Resolve paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, '..'))

    tables_dir = os.path.join(repo_root, 'results/tables')
    plots_dir = os.path.join(repo_root, 'results/plots')

    # Automatically ensure results directories exist before execution
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Prepare environment variables
    env = os.environ.copy()
    env["FORCE_SYNTHETIC"] = "true"

    # Append the experiments directory to PYTHONPATH for module resolution
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        env["PYTHONPATH"] = f"{current_dir}{os.pathsep}{current_pythonpath}"
    else:
        env["PYTHONPATH"] = current_dir

    print(f"Configured PYTHONPATH: {env['PYTHONPATH']}")
    print(f"Configured FORCE_SYNTHETIC: {env['FORCE_SYNTHETIC']}")

    # Order of execution for the benchmark suite
    scripts = [
        os.path.join(current_dir, 'resolution_test.py'),
        os.path.join(current_dir, 'precision_test.py'),
        os.path.join(current_dir, 'model_size_test.py'),
        os.path.join(current_dir, 'plot_results.py')
    ]

    for script in scripts:
        script_name = os.path.basename(script)
        print(f"\n========================================\nExecuting: {script_name}\n========================================")
        try:
            # Use sys.executable to maintain environment consistency
            result = subprocess.run(
                [sys.executable, script],
                env=env,
                check=True,
                cwd=repo_root
            )
            print(f"Finished: {script_name} successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failure in orchestrator while running {script_name}: {e}")
            sys.exit(e.returncode)

    print("\n========================================\nAll Benchmarks and Plot Generation Complete!\n========================================")

if __name__ == '__main__':
    main()
