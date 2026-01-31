# Cloud Deployment Guide: Deepnote & Camber

This guide explains how to deploy the **Phoenix Protocol (NeuroSnake)** research framework to cloud platforms for training on high-performance GPUs.

## 1. Deepnote Deployment

Deepnote is a collaborative data science notebook environment.

### Steps:
1.  **Create a Project:** Log in to [Deepnote](https://deepnote.com/) and create a new project.
2.  **Import Repository:**
    *   Click on the **Github** integration on the left sidebar.
    *   Connect your repository: `https://github.com/Vikaash-dev/Ai-research-paper-and-implementation-of-brain-tumor-detection-`
3.  **Hardware Selection:**
    *   For training, select a machine with a GPU (e.g., T4 or A10G) from the machine settings in the top right.
4.  **Open Notebook:**
    *   Open `Deepnote_Camber_Training.ipynb` in the file explorer.
5.  **Upload Data:**
    *   Upload your Br35H dataset zip file and unzip it into a `./data` folder.
    *   Structure: `./data/train`, `./data/validation`, `./data/test`.
6.  **Run:** Execute the notebook cells.

---

## 2. Camber Cloud Deployment

Camber provides managed compute environments for data science.

### Steps:
1.  **Launch Instance:**
    *   Log in to Camber Cloud.
    *   Launch a new JupyterLab instance with a GPU type (e.g., NVIDIA A100 or V100).
2.  **Clone Repository:**
    *   Open the terminal in JupyterLab.
    *   Run: `git clone https://github.com/Vikaash-dev/Ai-research-paper-and-implementation-of-brain-tumor-detection-.git`
    *   `cd Ai-research-paper-and-implementation-of-brain-tumor-detection-`
3.  **Environment Setup:**
    *   Run: `pip install -r requirements.txt`
4.  **Training:**
    *   Open `Deepnote_Camber_Training.ipynb`.
    *   Or run via terminal: 
        ```bash
        python src/train_phoenix.py --model-type neurosnake_liquid --epochs 100
        ```

## 3. Available Models

You can select the following architectures in the training script:

*   `neurosnake`: The original Phoenix Protocol (NeuroSnake-ViT).
*   `neurosnake_spectral`: Replaces MobileViT with **Spectral Gating** (Frequency Domain).
*   `neurosnake_kan`: Replaces MLP head with **Kolmogorov-Arnold Networks** (Splines).
*   `neurosnake_liquid`: Adds **Liquid Time-Constant (LTC)** dynamics for robustness.
*   `neurosnake_ttt`: (Experimental) Adds Test-Time Training capabilities.
