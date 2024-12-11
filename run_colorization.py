import os
import subprocess

PROGRESS_LOG = "progress.log"

def load_completed_scripts():
    if os.path.exists(PROGRESS_LOG):
        with open(PROGRESS_LOG, "r") as file:
            return set(file.read().splitlines())
    return set()

def save_completed_script(name):
    with open(PROGRESS_LOG, "a") as file:
        file.write(name + "\n")

def run_scripts(directory):
    scripts = sorted([f for f in os.listdir(directory) if f.endswith(".py")])

    completed_scripts = load_completed_scripts()

    for script in scripts:
        if script not in completed_scripts:
            script_path = os.path.join(directory, script)
            print(f"Script {script_path} is being run.")

            try:
                subprocess.run(["python", script_path], check=True)
                print(f"Script {script_path} is finished running.")
                save_completed_script(script)
            except subprocess.CalledProcessError as e:
                print(f"Error {e} occurred while running script {script_path}.")

if __name__ == "__main__":
    run_scripts("scripts/")
