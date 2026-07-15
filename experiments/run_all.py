import subprocess
import os
import sys

def run_script(script_path):
    print(f"--- Executing {script_path} ---")
    # Add experiments dir to PYTHONPATH so utils can be imported
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + env.get("PYTHONPATH", "")
    env["FORCE_SYNTHETIC"] = "true"  # Ensure it runs in headless environment

    result = subprocess.run([sys.executable, script_path], env=env)
    if result.returncode != 0:
        print(f"Error executing {script_path}")
    return result.returncode

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Ensure results directories exist
    os.makedirs(os.path.join(script_dir, "..", "results", "tables"), exist_ok=True)
    os.makedirs(os.path.join(script_dir, "..", "results", "plots"), exist_ok=True)

    scripts = [
        "resolution_test.py",
        "model_size_test.py",
        "precision_test.py",
        "plot_results.py"
    ]

    for script in scripts:
        full_path = os.path.join(script_dir, script)
        run_script(full_path)

    print("\nFull experiment suite completed.")

if __name__ == "__main__":
    main()
