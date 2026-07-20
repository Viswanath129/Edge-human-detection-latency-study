import os
import sys
import subprocess

def main():
    print("="*60)
    print("Starting Comprehensive Latency & Accuracy Trade-Off Benchmark Suite")
    print("="*60)

    # 1. Setup absolute paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))

    tables_dir = os.path.join(root_dir, "results", "tables")
    plots_dir = os.path.join(root_dir, "results", "plots")

    # Ensure required directories exist
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 2. Configure environment variables for the subprocesses
    env = os.environ.copy()
    env["FORCE_SYNTHETIC"] = "true"  # Force synthetic for headless validation

    # Append experiments folder to PYTHONPATH
    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        env["PYTHONPATH"] = f"{script_dir}{os.pathsep}{pythonpath}"
    else:
        env["PYTHONPATH"] = script_dir

    # 3. Define the tests to execute sequentially
    # Run the tests using sys.executable to maintain environment consistency
    test_scripts = [
        "resolution_test.py",
        "precision_test.py",
        "model_size_test.py"
    ]

    for script in test_scripts:
        script_path = os.path.join(script_dir, script)
        print(f"\n---> Executing benchmark: {script}")
        try:
            # We run subprocesses with sys.executable and ensure correct CWD
            result = subprocess.run(
                [sys.executable, script_path],
                env=env,
                cwd=root_dir,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print(f"Warnings/Stderr for {script}:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing {script}:")
            print(e.stderr)
            sys.exit(1)

    # 4. Run plotting script to update performance charts
    plotting_script = os.path.join(script_dir, "plot_results.py")
    print("\n---> Regenerating performance visualization plots...")
    try:
        result = subprocess.run(
            [sys.executable, plotting_script],
            env=env,
            cwd=root_dir,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error executing plot_results.py:")
        print(e.stderr)
        sys.exit(1)

    print("="*60)
    print("Benchmark Suite Execution Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
