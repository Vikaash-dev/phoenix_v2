import os
import subprocess
import json
import urllib.request
import urllib.error
# from runtime import DockerRuntime # Optional now

class ACI:
    """Agent Computer Interface: Simplifies interactions for the LLM. Supports both Docker and Local modes."""
    
    def __init__(self, runtime=None):
        self.runtime = runtime
        self.is_local = runtime is None

    def run_shell(self, command: str) -> str:
        """Executes a shell command."""
        if self.is_local:
            try:
                # Use subprocess for local execution
                result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
                return f"[SUCCESS]\n{result.decode()}"
            except subprocess.CalledProcessError as e:
                return f"[EXIT STATUS {e.returncode}]\n{e.output.decode()}"
        else:
            # Use Docker Runtime
            exit_code, output = self.runtime.exec_run(command)
            if exit_code == 0:
                return f"[SUCCESS]\n{output}"
            else:
                return f"[EXIT STATUS {exit_code}]\n{output}"

    def create_file(self, path: str, content: str) -> str:
        """Creates/Overwrites a file."""
        if self.is_local:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "w") as f:
                    f.write(content)
                return f"File {path} created."
            except Exception as e:
                return f"Failed to create {path}: {str(e)}"
        else:
            success = self.runtime.write_file(path, content)
            if success:
                return f"File {path} created."
            return f"Failed to create {path}."

    def read_file(self, path: str) -> str:
        """Reads a file."""
        if self.is_local:
            try:
                with open(path, "r") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"
        else:
            return self.runtime.read_file(path)

    def search_web(self, query: str) -> str:
        """Performs a web search using Tavily API (via urllib)."""
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            return "ERROR: TAVILY_API_KEY not found. Cannot search."
            
        url = "https://api.tavily.com/search"
        data = {
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "include_domains": ["github.com", "arxiv.org", "reddit.com", "stackoverflow.com"],
            "max_results": 5
        }
        
        try:
            req = urllib.request.Request(
                url, 
                data=json.dumps(data).encode('utf-8'), 
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                results = result.get("results", [])
                formatted = []
                for r in results:
                    formatted.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}...")
                return "\n\n".join(formatted)
        except Exception as e:
            return f"Search Failed: {str(e)}"

    def negative_search(self, query: str) -> str:
        """
        SUKUNA PROTOCOL: Mechanically enforced Falisfication.
        Appends critical keywords to find counter-evidence.
        """
        negative_query = f"{query} limitations failures problems issues benchmark vs sota"
        return f"Performing NEGATIVE ANALYSIS on: '{query}'\n" + self.search_web(negative_query)
