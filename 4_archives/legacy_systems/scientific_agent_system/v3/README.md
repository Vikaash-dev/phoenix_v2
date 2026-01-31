# V3: Dockerized Agent Runtime (Stdlib)

This is the SOTA architecture for the Automated Research System.
It runs the agent in a **Docker Container** using only the Python Standard Library (`subprocess`).

## Features

- **Security**: All code runs in a sandboxed container.
- **Persistence**: Maintains a shell session (conceptually).
- **No Dependencies**: No `pip install` required.

## Quick Start

```bash
# 1. Ensure Docker is running on your host
docker ps

# 2. Run the Agent Loop
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 agent_loop.py
```

## Architecture

- `runtime.py`: `DockerRuntime` class wrapping the `docker` CLI.
- `aci.py`: Agent-Computer Interface (simplified tools for LLM).
- `agent_loop.py`: The Main Loop (Observation -> Thought -> Action).
