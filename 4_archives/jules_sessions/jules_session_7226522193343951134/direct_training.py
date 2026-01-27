
import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jules_session_7226522193343951134.src.train_phoenix import train_phoenix_protocol

# Configuration
MODEL_TYPE = "neurosnake_vit"
EPOCHS = 10
BATCH_SIZE = 16
DATA_DIR = "./data"
OUTPUT_DIR = "./training_results"

if __name__ == "__main__":
    train_phoenix_protocol(
        data_dir=DATA_DIR,
        model_type=MODEL_TYPE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        output_dir=OUTPUT_DIR
    )
