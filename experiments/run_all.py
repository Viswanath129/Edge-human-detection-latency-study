import subprocess
import sys
import os

def main():
    # Ensure results directories exist
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    scripts = [
        "experiments/resolution_test.py",
        "experiments/precision_test.py",
        "experiments/model_size_test.py"
    ]

    for script in scripts:
        print(f"====================================================")
        print(f"Running {script}...")
        print(f"====================================================")
        result = subprocess.run([sys.executable, script], capture_output=True)
        if result.returncode != 0:
            print(f"Error running {script}:")
            print(result.stderr.decode())
        else:
            print(result.stdout.decode())

    # Finally, run the plotting script
    print("Generating plots...")
    subprocess.run([sys.executable, "results/plots/plot_results.py"])

if __name__ == "__main__":
    main()
