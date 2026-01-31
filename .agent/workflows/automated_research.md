---
description: Automated Research Agent Workflow
---
# Automated Scientific Research Workflow

This workflow automates the setup and execution of the Sakana-inspired multi-agent research system.

## 1. Prerequisites

Ensure you are in the project root: `/home/shadow_garden/brain-tumor-detection`

## 2. Environment Setup

// turbo

```bash
python3 -m venv scientific_agent_system/venv
source scientific_agent_system/venv/bin/activate
pip install -r scientific_agent_system/requirements.txt
```

## 3. Configuration (Optional)

If you require web search capabilities, export your Tavily API key:

```bash
export TAVILY_API_KEY="your-key-here"
```

## 4. Execution

Run the orchestrator with your research topic. The system will automatically:

1. **Adapt Skills**: Load relevant tools (e.g., medical imaging for tumor detection).
2. **Assemble Agents**: Create a team including Chief Scientist, ML Engineer, etc.
3. **Execute Research**: Formulate hypotheses, run experiments (simulated), and report findings.

```bash
# Example: Brain Tumor Research
python3 scientific_agent_system/orchestrator.py --topic "optimizing cnn for brain tumor mri segmentation"
```

## 5. Output

The agents will output their conversation and findings to the console (and eventually to log files).
