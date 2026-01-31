import json
import os
import urllib.request
import urllib.error

class LLMClient:
    """
    Zero-dependency LLM Client using standard library.
    Supports OpenAI-compatible APIs (OpenAI, DeepSeek, vLLM).
    """
    def __init__(self, model="gpt-4o"):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model

    def chat_completion(self, messages, temperature=0.0):
        if not self.api_key:
            # Fallback for verification/demo if no key provided
            print("⚠️  OPENAI_API_KEY not set. Using Mock Response for verification.")
            return self._mock_fallback(messages)

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        try:
            req = urllib.request.Request(
                url, 
                data=json.dumps(data).encode('utf-8'), 
                headers=headers
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode()
            return f"Error calling LLM: {e.code} - {err_body}"
        except Exception as e:
            return f"Error: {str(e)}"

    def _mock_fallback(self, messages):
        """Simulates Sukuna Intelligence for testing flow without cost."""
        history_len = len(messages)
        
        # Turn 1: Initialization & State Update
        if history_len <= 3:
            return """THOUGHT: I need to initialize the agent state. I will create the initial `agent.md` with the inferred goal and research strategy.
PLAN: 1. Create agent.md.
ACTION: create_file('scientific_agent_system/v3/agent.md', '# Sukuna Agent State\\n\\n## 1. Meta-State\\n- Status: PLAN_ESTABLISHED\\n\\n## 2. Strategy\\n- [ ] Research Phoenix Mamba V2\\n- [ ] Negative Analysis: Check gradient stability.\\n\\n## 3. Knowledge Graph\\n(Empty)')"""
        
        # Turn 2: Execution (Retrieval)
        elif history_len <= 5:
            return """THOUGHT: State is initialized. Now I must execute the first step of the strategy: Research Phoenix Mamba V2.
PLAN: 1. Search web for the specific architecture.
ACTION: search_web('Phoenix Mamba V2 architecture github')"""
            
        # Turn 3: Completion
        else:
            return "THOUGHT: I have verified the agent loop mechanics. I will now stop.\nPLAN: Stop.\nACTION: stop('Verification complete')"
