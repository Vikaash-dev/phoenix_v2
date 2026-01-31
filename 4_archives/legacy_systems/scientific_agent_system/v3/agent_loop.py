import argparse
import sys
import os
import re
from aci import ACI
from llm_client import LLMClient
from prompts import SYSTEM_PROMPT

def parse_action(llm_output):
    """Parses the LLM output for function calls using regex."""
    # Pattern for create_file: ACTION: create_file('path', 'content')
    create_file_pattern = r"ACTION:\s*create_file\('((?:[^'\\]|\\.)*)',\s*'((?:[^'\\]|\\.)*)'\)"
    match_create = re.search(create_file_pattern, llm_output, re.DOTALL)
    if match_create:
        path = match_create.group(1).replace("\\'", "'")
        content = match_create.group(2).replace("\\'", "'")
        return "create_file", (path, content)

    # Pattern for single argument actions
    single_arg_pattern = r"ACTION:\s*(run_shell|search_web|negative_search)\('((?:[^'\\]|\\.)*)'\)"
    match_single = re.search(single_arg_pattern, llm_output, re.DOTALL)
    if match_single:
        action = match_single.group(1)
        arg = match_single.group(2).replace("\\'", "'")
        return action, arg

    if "ACTION: stop" in llm_output:
        return "stop", None

    # Fallback to check simple keywords if explicit parsing failed
    if "step" in llm_output.lower() or "thought" in llm_output.lower():
         return "reasoning", None # Just thought, no action
    return "unknown", None

def run_agent_loop(topic: str, max_steps: int = 15, state_file: str = "scientific_agent_system/v3/agent.md", model: str = "gpt-4o"):
    print(f"🚀 Starting Sukuna Agent Loop (V6) for: {topic} (Model: {model})")

    # 1. Setup
    llm = LLMClient(model=model)
    aci = ACI() # Defaults to Local Mode
    agent_md_path = state_file

    # Initialize basic agent.md if missing
    if not os.path.exists(agent_md_path):
        try:
             # Ensure directory exists
             os.makedirs(os.path.dirname(os.path.abspath(agent_md_path)), exist_ok=True)
             with open(agent_md_path, "w") as f:
                 f.write(f"# Sukuna Agent State\nTopic: {topic}\nStatus: INITIALIZING")
        except Exception as e:
            print(f"⚠️ Could not create state file: {e}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"BEGIN SUKUNA PROTOCOL. Topic: {topic}"}
    ]

    try:
        for step in range(max_steps):
            print(f"\n--- STEP {step + 1} ---")

            # 2. Inject State (The Brain)
            try:
                with open(agent_md_path, "r") as f:
                    state_content = f.read()
                # Append context to ensure visibility
                context_msg = f"\n\n[CURRENT AGENT.MD STATE]:\n{state_content}\n\n[INSTRUCTION]: Read state -> Think -> Act (Search/Update State)."
                if messages[-1]["role"] == "user":
                     messages[-1]["content"] += context_msg
                else:
                     messages.append({"role": "user", "content": context_msg})
            except Exception as e:
                 print(f"⚠️ Could not read agent.md: {e}")

            # 3. Think
            response = llm.chat_completion(messages)
            print(f"🤖 AGENT:\n{response}")
            messages.append({"role": "assistant", "content": response})

            # 4. Act
            action_type, args = parse_action(response)

            observation = ""
            if action_type == "run_shell":
                observation = aci.run_shell(args)
            elif action_type == "search_web":
                observation = aci.search_web(args)
            elif action_type == "negative_search":
                observation = aci.negative_search(args)
            elif action_type == "create_file":
                observation = aci.create_file(args[0], args[1])
            elif action_type == "stop":
                print("✅ Agent requested STOP.")
                break
            elif action_type == "reasoning":
                observation = "Thinking..."
            else:
                observation = "No valid ACTION found. Please use: run_shell, search_web, or create_file."

            print(f"👁️ OBSERVATION:\n{observation[:500]}..." if len(observation) > 500 else f"👁️ OBSERVATION:\n{observation}")
            messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

    except Exception as e:
        print(f"❌ Critical Error in Agent Loop: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default="Research Sukuna Protocol")
    parser.add_argument("--state-file", type=str, default="scientific_agent_system/v3/agent.md", help="Path to the agent state file")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model to use")
    args = parser.parse_args()
    run_agent_loop(args.topic, state_file=args.state_file, model=args.model)

if __name__ == "__main__":
    main()
