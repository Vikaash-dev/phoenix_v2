
import os
import zipfile
import glob

def extract_jules_sessions():
    base_dir = "/home/shadow_garden/brain-tumor-detection"
    output_base = os.path.join(base_dir, "extracted_jules")
    
    # Get all jules zip files
    zip_files = glob.glob(os.path.join(base_dir, "jules_session_*.zip"))
    
    if not zip_files:
        print("No jules_session zip files found.")
        return

    print(f"Found {len(zip_files)} zip files.")
    
    for zip_path in zip_files:
        filename = os.path.basename(zip_path)
        # Create a folder name based on the zip file (stripping .zip)
        folder_name = filename.replace('.zip', '')
        extract_path = os.path.join(output_base, folder_name)
        
        print(f"Extracting {filename} to {extract_path}...")
        
        os.makedirs(extract_path, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print("Done.")
        except Exception as e:
            print(f"Error extracting {filename}: {e}")

if __name__ == "__main__":
    extract_jules_sessions()
