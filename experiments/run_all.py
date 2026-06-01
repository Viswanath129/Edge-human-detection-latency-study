import os
import subprocess
import sys

def run_all_benchmarks():
    scripts = [
        "resolution_test.py",
        "precision_test.py",
        "model_size_test.py"
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Set environment to force synthetic if needed or just pass it
    env = os.environ.copy()
    if "FORCE_SYNTHETIC" not in env:
        env["FORCE_SYNTHETIC"] = "true"

    for script in scripts:
        script_path = os.path.join(base_dir, script)
        print(f"========================================")
        print(f"Running {script}...")
        print(f"========================================")

        try:
            # Using sys.executable to ensure we use the same python interpreter
            subprocess.run([sys.executable, script_path], check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")

if __name__ == "__main__":
    run_all_benchmarks()
