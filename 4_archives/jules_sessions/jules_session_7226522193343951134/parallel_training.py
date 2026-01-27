
import subprocess
import os

# Configuration
MODEL_TYPE = "neurosnake_vit"
EPOCHS = 10
BATCH_SIZE = 16
DATA_DIR = "./data"
OUTPUT_DIR = "./training_results"
NUM_INSTANCES = 3

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create a list of commands to run
commands = []
for i in range(NUM_INSTANCES):
    instance_output_dir = os.path.join(OUTPUT_DIR, f"instance_{i}")
    os.makedirs(instance_output_dir, exist_ok=True)
    log_file = os.path.join(instance_output_dir, "training.log")
    command = (
        f"python jules_session_7226522193343951134/src/train_phoenix.py "
        f"--model-type {MODEL_TYPE} "
        f"--epochs {EPOCHS} "
        f"--batch-size {BATCH_SIZE} "
        f"--data-dir {DATA_DIR} "
        f"--output-dir {instance_output_dir} "
        f"> {log_file} 2>&1"
    )
    commands.append(command)

# Execute the commands in parallel
processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]

# Wait for all processes to complete
for p in processes:
    p.wait()

print("Parallel training complete.")
