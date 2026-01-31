import os
from typing import List, Dict, Any, Callable
import json

class Skill:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def execute(self, *args, **kwargs):
        print(f"Executing Skill: {self.name}...")
        return self.func(*args, **kwargs)

class SkillRegistry:
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self._register_default_skills()

    def register_skill(self, name: str, description: str, func: Callable):
        self.skills[name] = Skill(name, description, func)

    def get_skill(self, name: str) -> Skill:
        return self.skills.get(name)

    def list_skills(self) -> str:
        return "\n".join([f"- {s.name}: {s.description}" for s in self.skills.values()])

    def _register_default_skills(self):
        # Tavily Search (Placeholder if API not present, or using Brave search via MCP if requested)
        # Since the user requested "Tavily", we'll attempt to add a specific skill for it.
        self.register_skill("web_search", "Search the web for scientific information", self._tavily_search_skill)
        self.register_skill("read_file", "Read file content", self._read_file_skill)
        self.register_skill("write_file", "Write content to a file", self._write_file_skill)
        self.register_skill("execute_python", "Execute Python code", self._execute_python_skill)

    def _tavily_search_skill(self, query: str) -> str:
        # Check for API Key
        api_key = os.environ.get("TAVILY_API_KEY", "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")
        if not api_key:
            return f"ERROR: TAVILY_API_KEY not found. Simulating search for: {query}"
        
        try:
            # Use requests to properly call Tavily
            import requests
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": 3
            }
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                results = response.json().get("results", [])
                formatted = "\n".join([f"- [{r['title']}]({r['url']}): {r['content'][:200]}..." for r in results])
                return f"Sources:\n{formatted}"
            else:
                return f"Tavily API Error: {response.status_code} - {response.text}"
        except ImportError:
             return f"ERROR: 'requests' library not found. Please install it using: pip install requests. Placeholder: {query}"
        except Exception as e:
            return f"Error performing search: {str(e)}"

    def _read_file_skill(self, file_path: str) -> str:
        try:
            with open(file_path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"

    def _write_file_skill(self, file_path: str, content: str) -> str:
        try:
            with open(file_path, "w") as f:
                f.write(content)
            return f"Successfully wrote to {file_path}"
        except Exception as e:
            return f"Error writing to {file_path}: {str(e)}"

    def _execute_python_skill(self, code: str) -> str:
        # VERY DANGEROUS in production, but standard for "AI Scientist" prototypes
        # We will capture stdout
        import io
        import sys
        
        # Create a captured stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            exec(code, globals())
            output = captured_output.getvalue()
            return f"Execution Output:\n{output}"
        except Exception as e:
            return f"Execution Error: {str(e)}"
        finally:
            sys.stdout = sys.__stdout__

    def adapt_skills_for_task(self, task_description: str):
        """
        Dynamically adjusts or suggests new skills based on the task.
        In a full implementation, this might load plugins.
        """
        print(f"Adapting skills for task: {task_description}")
        if "brain tumor" in task_description.lower():
            self.register_skill("medical_imaging_analysis", "Analyze MRI/medical images (mock)", lambda x: "Analyzing Medical Image...")
            print(" -> Added specialized skill: medical_imaging_analysis")
        if "paper" in task_description.lower():
             self.register_skill("generate_citation", "Generate BibTeX citations", lambda x: "Citation...")
             print(" -> Added specialized skill: generate_citation")

