# Cloud Deployment Guide: Deepnote & Camber

This guide explains how to train the Phoenix Protocol (NeuroKAN) on cloud platforms using the provided orchestrator notebook.

## Prerequisites

1.  **Kaggle Account**: You need an API key to download the dataset.
    *   Go to Kaggle -> Settings -> API -> Create New Token (`kaggle.json`).
    *   Note the `username` and `key`.

## Deepnote Setup

1.  **Create Project**: Upload this repository to Deepnote.
2.  **Set Environment Variables**:
    *   Go to "Integrations" or "Environment Variables".
    *   Add `KAGGLE_USERNAME`: Your username.
    *   Add `KAGGLE_KEY`: Your API key.
3.  **Run**: Open `Deepnote_Camber_Training.ipynb` and run all cells.

## Camber Cloud Setup

1.  **Start Engine**: Launch a GPU instance (e.g., V100/A100).
2.  **Upload Code**: Clone the repo.
3.  **Set Secrets**:
    *   In the terminal: `export KAGGLE_USERNAME=...` and `export KAGGLE_KEY=...`
    *   Or add them to your `.bashrc`.
4.  **Run**:
    *   Interactive: Open the Jupyter Notebook.
    *   Headless: Convert the notebook to script or run individual commands:
        ```bash
        pip install -r requirements.txt
        python tools/setup_medical_data.py
        python src/train_phoenix.py --model-type neurokan --mixed-precision
        ```

## Troubleshooting

*   **Dataset Download Fail**: Ensure API keys are correct. If Kaggle CLI fails, try manual download and upload to `data/raw`.
*   **OOM (Out of Memory)**: Reduce batch size in `src/train_phoenix.py` (argument `--batch-size 16`).
*   **Import Errors**: Ensure you are running from the repository root so `src` and `models` are discoverable.
