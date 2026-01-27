
import papermill as pm
import os

# Configuration
INPUT_NOTEBOOK = "jules_session_7226522193343951134/Deepnote_Camber_Training.ipynb"
OUTPUT_NOTEBOOK = "jules_session_7226522193343951134/Deepnote_Camber_Training_output.ipynb"
MODEL_TYPE = "neurosnake_vit"
EPOCHS = 10
BATCH_SIZE = 16
DATA_DIR = "./data"
OUTPUT_DIR = "./training_results"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Execute the notebook with parameters
pm.execute_notebook(
    INPUT_NOTEBOOK,
    OUTPUT_NOTEBOOK,
    parameters=dict(
        MODEL_TYPE=MODEL_TYPE,
        EPOCHS=EPOCHS,
        BATCH_SIZE=BATCH_SIZE,
        DATA_DIR=DATA_DIR,
        OUTPUT_DIR=OUTPUT_DIR
    )
)

print("Notebook execution complete.")
