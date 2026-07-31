import os
import sys
import subprocess

def main():
    print("=========================================")
    print("YOLOv8 Edge Benchmarking Automation Suite")
    print("=========================================")

    # 1. Ensure directory structures exist before executing
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tables_dir = os.path.abspath(os.path.join(script_dir, "../results/tables"))
    plots_dir = os.path.abspath(os.path.join(script_dir, "../results/plots"))
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Verified directory structures: {tables_dir} and {plots_dir}")

    # 2. Configure environment with FORCE_SYNTHETIC=true and updated PYTHONPATH
    env = os.environ.copy()
    env["FORCE_SYNTHETIC"] = "true"

    experiments_dir = os.path.abspath(script_dir)
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        env["PYTHONPATH"] = experiments_dir + os.pathsep + existing_pythonpath
    else:
        env["PYTHONPATH"] = experiments_dir
    print(f"Configured PYTHONPATH: {env['PYTHONPATH']}")
    print("Set FORCE_SYNTHETIC=true to force synthetic frame generation in headless environment.")

    # 3. Execute individual test scripts in sequence
    scripts = ["resolution_test.py", "precision_test.py", "model_size_test.py", "plot_results.py"]

    for script in scripts:
        script_path = os.path.join(experiments_dir, script)
        print(f"\n>>> Executing script: {script} ...")

        # Use sys.executable to ensure environment consistency
        cmd = [sys.executable, script_path]
        try:
            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Warnings/Errors:", result.stderr, file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing {script}:", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
            sys.exit(e.returncode)

    print("\n=========================================")
    print("Benchmark Suite Execution Completed Successfully!")
    print("=========================================")

if __name__ == "__main__":
    main()
