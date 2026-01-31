import sys
import os
import argparse
from crew import ScientificResearchCrew

def main():
    parser = argparse.ArgumentParser(description="Sakana-style Production Level Research System (CrewAI)")
    parser.add_argument("--topic", type=str, required=True, help="Research topic or problem statement")
    args = parser.parse_args()

    print(f"🚀 Starting Automated Research Crew")
    print(f"Target Topic: {args.topic}")
    print("==================================================")
    
    # Check for API Key
    if not os.environ.get("TAVILY_API_KEY"):
        print("⚠️  WARNING: TAVILY_API_KEY not found. Web search will fail.")

    crew = ScientificResearchCrew(args.topic)
    result = crew.run()
    
    print("\n\n########################")
    print("## FINAL RESEARCH PAPER ##")
    print("########################\n")
    print(result)
    
    # Save to file
    with open("research_paper_crewai.md", "w") as f:
        f.write(str(result))
    print("\n✅ Saved to research_paper_crewai.md")

if __name__ == "__main__":
    main()
