import subprocess
import os

def run_experiment(script_name):
    print(f"--- Running {script_name} ---")
    env = os.environ.copy()
    # Ensure experiments directory is in PYTHONPATH for utils import
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + ":" + os.path.dirname(os.path.abspath(__file__))

    result = subprocess.run(["python3", os.path.join("experiments", script_name)],
                            env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Errors in {script_name}:")
        print(result.stderr)

def main():
    experiments = [
        "resolution_test.py",
        "precision_test.py",
        "model_size_test.py"
    ]

    # Force synthetic for headless environment
    os.environ["FORCE_SYNTHETIC"] = "true"

    for exp in experiments:
        run_experiment(exp)

    print("All experiments completed.")

if __name__ == "__main__":
    main()
