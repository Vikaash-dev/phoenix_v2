from typing import List, Any
import time

class Agent:
    def __init__(self, role,RE, goal, backstory, tools=None, verbose=True, memory=True):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.verbose = verbose

    def execute_task(self, task):
        print(f"\n[{self.role}] Working on task: {task.description[:50]}...")
        # Simulate thinking
        time.sleep(0.5)
        # In a real system, this calls LLM. Here we mock or use tools.
        if self.tools:
            for tool in self.tools:
                try:
                    # Heuristic: if tool is Tavily, use it for "search" tasks
                    if "search" in task.description.lower() and "tavily" in tool.name.lower():
                        print(f"[{self.role}] Using tool: {tool.name}")
                        return tool.func("brain tumor detection mri latest research")
                except:
                    pass
        return f"[Output from {self.role}]: Completed {task.description[:30]}..."

class Task:
    def __init__(self, description, expected_output, agent, context=None):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.context = context
        self.output = None

    def execute_sync(self):
        self.output = self.agent.execute_task(self)
        return self.output

class Process:
    sequential = "sequential"

class Crew:
    def __init__(self, agents, tasks, process=Process.sequential, verbose=True):
        self.agents = agents
        self.tasks = tasks
        self.process = process
        self.verbose = verbose

    def kickoff(self):
        print("Crew kickoff! Agents are working...")
        results = []
        for task in self.tasks:
            res = task.execute_sync()
            results.append(res)
            # Pass output as context to next
            # Note: simplified for mock
        
        final_summary = "\n\n".join([str(r) for r in results])
        return final_summary
