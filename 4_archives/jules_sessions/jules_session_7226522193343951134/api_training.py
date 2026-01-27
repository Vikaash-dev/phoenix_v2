
import os
import requests
import json
import time

# --- Configuration ---
API_KEY = "26fa0d8c9b75842fc19430ba6061977146e1d568bb7c1a13af0c0008bc514758ae245488e057fd91c4be3fe164f3efcf7dc183406e7db980de31dc902df999fa" 
PROJECT_ID = "462841f8-eeb7-4df1-ae17-c4d16f4610ff" 
NOTEBOOK_ID = "791731d6c4f64edd92cbce4d8ad88eaf" 
OUTPUT_DIR = "./training_results"

# --- Helper Functions ---
def get_api_headers():
    if not API_KEY:
        raise ValueError("DEEPNOTE_API_KEY environment variable not set.")
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

def execute_notebook(project_id, notebook_id):
    """Triggers the execution of a Deepnote notebook."""
    url = f"https://api.deepnote.com/v1/projects/{project_id}/notebooks/{notebook_id}/execute"
    try:
        response = requests.post(url, headers=get_api_headers())
        response.raise_for_status()
        print("Notebook execution started successfully.")
        # Check if the response is not empty before trying to decode it
        if response.text:
            return response.json()
        else:
            return {}
    except requests.exceptions.RequestException as e:
        print(f"Error executing notebook: {e}")
        if e.response:
            print(f"Response content: {e.response.text}")
        return None

def get_notebook_status(project_id, notebook_id):
    """Retrieves the status of a notebook."""
    url = f"https://api.deepnote.com/v1/projects/{project_id}/notebooks/{notebook_id}"
    try:
        response = requests.get(url, headers=get_api_headers())
        response.raise_for_status()
        return response.json().get("status")
    except requests.exceptions.RequestException as e:
        print(f"Error getting notebook status: {e}")
        return None

def download_file(project_id, path, destination):
    """Downloads a file from the Deepnote project."""
    url = f"https://api.deepnote.com/v1/projects/{project_id}/files?path={path}"
    try:
        response = requests.get(url, headers=get_api_headers(), stream=True)
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {path} to {destination}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Execute the notebook
    execution_info = execute_notebook(PROJECT_ID, NOTEBOOK_ID)
    if not execution_info:
        exit()

    # 2. Monitor the execution
    while True:
        status = get_notebook_status(PROJECT_ID, NOTEBOOK_ID)
        if status in ["running", "queued"]:
            print(f"Notebook status: {status}. Waiting...")
            time.sleep(30)
        elif status == "finished":
            print("Notebook execution finished.")
            break
        elif status in ["error", "interrupted"]:
            print(f"Notebook execution failed with status: {status}")
            exit()
        else:
            print(f"Unknown notebook status: {status}")
            exit()

    # 3. Download the results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Example: download the trained model and logs
    # Note: You'll need to know the exact paths in your Deepnote project
    model_path = "training_results/neurosnake_vit_best.h5" 
    log_path = "training_results/training_log.csv"
    
    download_file(PROJECT_ID, model_path, os.path.join(OUTPUT_DIR, "neurosnake_vit_best.h5"))
    download_file(PROJECT_ID, log_path, os.path.join(OUTPUT_DIR, "training_log.csv"))

    print("Training complete and results downloaded.")
