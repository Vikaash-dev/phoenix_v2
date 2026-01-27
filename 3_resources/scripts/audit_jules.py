
import os
import ast
import glob

def audit():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Auditing relative to: {base_dir}")
    
    # Pattern to find all python files in jules_session directories
    # We look for jules_session_* folders
    jules_dirs = glob.glob(os.path.join(base_dir, "jules_session_*"))
    
    total_files = 0
    total_errors = 0
    
    for j_dir in jules_dirs:
        print(f"Scanning {os.path.basename(j_dir)}...")
        for root, _, files in os.walk(j_dir):
            for file in files:
                if file.endswith(".py"):
                    total_files += 1
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            tree = ast.parse(f.read())
                    except SyntaxError as e:
                        print(f"❌ Syntax Error in {os.path.relpath(path, base_dir)}: {e}")
                        total_errors += 1
                    except Exception as e:
                        print(f"⚠️ Error reading {os.path.relpath(path, base_dir)}: {e}")
                        total_errors += 1

    print(f"\nAudit Report: Scanned {total_files} files. Found {total_errors} errors.")

if __name__ == "__main__":
    audit()
