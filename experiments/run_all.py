import os
import sys
import subprocess

def run_script(script_name):
    print(f"\n{'='*20}")
    print(f"Running {script_name}...")
    print(f"{'='*20}")

    # Add current directory to PYTHONPATH so scripts can find utils.py
    env = os.environ.copy()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = current_dir + os.pathsep + env.get("PYTHONPATH", "")
    env["FORCE_SYNTHETIC"] = "true"  # Ensure headless-safe execution

    result = subprocess.run([sys.executable, os.path.join(current_dir, script_name)], env=env)
    if result.returncode != 0:
        print(f"Error running {script_name}")
        return False
    return True

def main():
    # Ensure directories exist
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(base_dir, "results", "tables"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results", "plots"), exist_ok=True)

    scripts = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py",
        "plot_results.py"
    ]

    for script in scripts:
        if not run_script(script):
            sys.exit(1)

    print("\nAll benchmarks and plots completed successfully.")

if __name__ == "__main__":
    main()
