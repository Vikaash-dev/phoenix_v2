import argparse
import sys
import os
import time
from communication import CommunicationChannel
from skills import SkillRegistry
from agents import AgentFactory

def main():
    parser = argparse.ArgumentParser(description="Sakana-style Production Level Research System")
    parser.add_argument("--topic", type=str, required=True, help="Research topic or problem statement")
    parser.add_argument("--mode", type=str, default="production", help="Mode: standard or production")
    parser.add_argument("--iterations", type=int, default=1, help="Number of refinement loops")
    args = parser.parse_args()

    print(f"🚀 Starting Research System [Mode: {args.mode}]")
    print(f"Target Topic: {args.topic}")
    print("==================================================")

    # 1. Setup
    comms = CommunicationChannel()
    skills = SkillRegistry()
    skills.adapt_skills_for_task(args.topic)
    team = AgentFactory.create_team(args.topic, comms, skills)

    # 2. Phase 1: Idea Generation
    print("\n--- Phase 1: Idea Generation & Decomposition ---")
    chief = team["ChiefScientist"]
    hypothesis = chief.formulate_hypothesis(args.topic)
    
    architect = team.get("Architect")
    if architect:
        # Architect breaks it down
        if not comms.get_history():
            # Artificial trigger for the Architect if no prior message exists
            from communication import Message
            trigger_msg = Message("System", "Architect", f"Please decompose the hypothesis: {hypothesis}", "instruction")
            architect.process_message(trigger_msg)
        else:
            architect.process_message(comms.get_history()[-1])
        architect.decompose_approach(hypothesis)
    
    # 3. Phase 2: Parallel Refinement & Negative Analysis
    print("\n--- Phase 2: Parallel Discussion & Negative Analysis ---")
    # Simulate a round-table discussion
    discussion_rounds = 3
    for r in range(discussion_rounds):
        print(f"\n[Discussion Round {r+1}]")
        
        # Negative Analyst critiques whatever is on the table
        neg_analyst = team.get("NegativeAnalyst")
        if neg_analyst:
            # Look at recent broadcasts
            recent_msgs = comms.get_history()
            if recent_msgs:
                last_significant = recent_msgs[-1]
                neg_analyst.critique(last_significant.content)
        
        # Chief Scientist or Architect responds
        if architect:
            # Architect refines based on critique
            architect.broadcast(f"Refining Block {r+1} based on critique...", "refinement")

    # 4. Phase 3: Execution
    print("\n--- Phase 3: Execution ---")
    engineer = team.get("MLEngineer")
    if engineer:
        engineer.execute_experiment("Refined Plan from Phase 2")

    # 5. Phase 4: Evaluation & Synthesis
    print("\n--- Phase 4: Cross-Evaluation & Synthesis ---")
    reviewer = team.get("Reviewer")
    final_output = "No report generated."
    
    if reviewer:
        # Evaluate the Engineer's report
        engineer_msgs = comms.get_messages_for("ChiefScientist") # Engineer reported to Chief
        if engineer_msgs:
            last_report = engineer_msgs[-1].content
            evaluation = reviewer.evaluate(last_report)
            
            # Synthesize Final Paper
            final_output = f"""# Automated Research Paper: {args.topic}
            
## 1. Abstract
{hypothesis}

## 2. Methodology (Functional Decomposition)
{architect.memory[-1].content if architect and architect.memory else "See decomposition logs"}

## 3. Experimental Results
{last_report}

## 4. Discussion & Negative Analysis
{comms.get_messages_for("ALL")[-1].content if comms.get_messages_for("ALL") else "No negative analysis recorded"}

## 5. Peer Review & Conclusion
{evaluation}
"""
            
    # Write to file
    filename = f"research_paper_{int(time.time())}.md"
    with open(filename, "w") as f:
        f.write(final_output)
    
    print(f"\n✅ Production Research Session Complete. Paper saved to: {filename}")

if __name__ == "__main__":
    main()
