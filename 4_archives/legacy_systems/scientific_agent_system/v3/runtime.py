import subprocess
import os
import time

class DockerRuntime:
    """
    Robust, Dependency-Free Docker Runtime.
    Uses 'subprocess' to call the docker CLI directly.
    """
    def __init__(self, image="python:3.9-slim", workspace_dir="/workspace"):
        self.image = image
        self.workspace_dir = workspace_dir
        self.container_name = f"agent_v3_{int(time.time())}"

    def _run_cmd(self, cmd_list):
        """Helper to run shell commands."""
        try:
            result = subprocess.run(
                cmd_list, 
                capture_output=True, 
                text=True, 
                check=False
            )
            return result.returncode, result.stdout + result.stderr
        except FileNotFoundError:
            return 127, "Error: 'docker' command not found."

    def start(self):
        """Starts a detached container."""
        print(f"🐳 Starting Docker Container ({self.image})...")
        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-v", f"{os.getcwd()}:{self.workspace_dir}",
            "-w", self.workspace_dir,
            self.image,
            "tail", "-f", "/dev/null"
        ]
        code, out = self._run_cmd(cmd)
        if code != 0:
            raise RuntimeError(f"Docker start failed: {out}")
        
        # Provision basic tools strictly inside the container
        self.exec_run("apt-get update && apt-get install -y git curl", timeout=60)
        print(f"✅ Container {self.container_name} started.")

    def stop(self):
        print(f"🛑 Stopping container {self.container_name}...")
        self._run_cmd(["docker", "rm", "-f", self.container_name])

    def exec_run(self, cmd_str, timeout=10):
        """Executes a command inside the container."""
        # We wrap in bash -c to handle pipes/redirects
        full_cmd = [
            "docker", "exec", 
            self.container_name, 
            "bash", "-c", cmd_str
        ]
        # In a real agent, we'd handle timeout loop here
        code, out = self._run_cmd(full_cmd)
        return code, out

    def write_file(self, file_path, content):
        """Writes file using cat and heredoc to avoid escaping hell."""
        # EOF technique to safely write multiline strings
        heredoc_cmd = f"cat << 'EOF' > {file_path}\n{content}\nEOF"
        code, out = self.exec_run(heredoc_cmd)
        return code == 0

    def read_file(self, file_path):
        code, out = self.exec_run(f"cat {file_path}")
        return out
