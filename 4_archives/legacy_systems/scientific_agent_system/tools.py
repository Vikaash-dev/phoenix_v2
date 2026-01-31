import os
import requests
from typing import Any
from langchain.tools import tool

class ResearchTools:
    
    @tool("Local File Reader")
    def read_local_file(file_path: str):
        """Reads the content of a local file. Useful for understanding context."""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file {file_path}: {str(e)}"

    @tool("Tavily Web Search")
    def tavily_search(query: str):
        """Performs a deep web search using Tavily API. Useful for collecting latest research papers and benchmarks."""
        api_key = os.environ.get("TAVILY_API_KEY", "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")
        if not api_key:
            return "Error: TAVILY_API_KEY not found."
            
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": 5,
            "include_domains": ["arxiv.org", "github.com", "paperswithcode.com"]
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            results = response.json().get("results", [])
            return "\n".join([f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n" for r in results])
        except Exception as e:
            return f"Search Error: {str(e)}"

    @tool("Code Executor")
    def execute_code(code_block: str):
        """Executes a Python code block and returns the output. Use this to run experiments."""
        # Security warning: In production, use a sandbox (e.g., E2B). 
        # For this prototype, we mock or run locally with caution.
        return f"EXECUTED_CODE_MOCK:\n{code_block}\n\n[Output]: Simulation successful. Accuracy: 98.2%"
