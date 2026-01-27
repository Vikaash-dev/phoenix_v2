
import os
import requests
import json
import time

# --- Configuration ---
API_KEY = "26fa0d8c9b75842fc19430ba6061977146e1d568bb7c1a13af0c0008bc514758ae245488e057fd91c4be3fe164f3efcf7dc183406e7db980de31dc902df999fa"
PROJECT_ID = "462841f8-eeb7-4df1-ae17-c4d16f4610ff"
NOTEBOOK_ID = "791731d6c4f64edd92cbce4d8ad88eaf"

# --- Helper Functions ---
def get_api_headers():
    if not API_KEY:
        raise ValueError("DEEPNOTE_API_KEY environment variable not set.")
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

def get_notebook_status(project_id, notebook_id):
    """Retrieves the status of a notebook."""
    url = f"https://api.deepnote.com/v1/projects/{project_id}/notebooks/{notebook_id}"
    try:
        response = requests.get(url, headers=get_api_headers())
        response.raise_for_status()
        return response.json().get("status")
    except requests.exceptions.RequestException as e:
        print(f"Error getting notebook status: {e}")
        if e.response:
            print(f"Response content: {e.response.text}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    status = get_notebook_status(PROJECT_ID, NOTEBOOK_ID)
    if status:
        print(f"Notebook status: {status}")
    else:
        print("Could not retrieve notebook status.")
