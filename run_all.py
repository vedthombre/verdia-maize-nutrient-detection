import subprocess
import sys

def run_script(script_name):
    print(f"\\n--- Running {script_name} ---")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"Error running {script_name}")
        sys.exit(result.returncode)

run_script("train_expA_baseline.py")
run_script("train_expB_augmented.py")
run_script("evaluate_experiments.py")
print("\\nAll phases completed successfully.")
