from crewai import Agent
from tools import ResearchTools

class ScientificAgents:
    def chief_investigator(self):
        return Agent(
            role='Chief Investigator',
            goal='Orchestrate the entire research lifecycle and ensure novelty.',
            backstory='Renowned scientist with a track record of breakthrough discoveries. You manage the team and ensure the research is rigorous.',
            tools=[ResearchTools.tavily_search, ResearchTools.read_local_file],
            verbose=True,
            memory=True
        )

    def methodology_architect(self):
        return Agent(
            role='Methodology Architect',
            goal='Decompose research problems into functional blocks and architectural diagrams.',
            backstory='Expert systems architect. You engage in "Paper2Code" analysis, breaking down abstract ideas into implementable components.',
            tools=[ResearchTools.read_local_file],
            verbose=True,
            memory=True
        )

    def experimental_engineer(self):
        return Agent(
            role='Experimental Engineer',
            goal='Implement the architectural plan into executable Python code.',
            backstory='Senior ML Engineer. You turn designs into working code. You prioritize clean, modular, and error-free implementation.',
            tools=[ResearchTools.execute_code],
            verbose=True,
            memory=True
        )

    def scientific_critic(self):
        return Agent(
            role='Scientific Critic',
            goal='Perform negative analysis and "Reflexion" on proposed plans.',
            backstory='A rigorous peer reviewer. You actively look for flaws, edge cases, and logical fallacies. You are the "Negative Analyst".',
            verbose=True,
            memory=True
        )

    def security_auditor(self):
        return Agent(
            role='Security Auditor',
            goal='Audit generated code/plans for security vulnerabilities and logic errors.',
            backstory='Inspired by DeepCode. You ensure the safety and robustness of the implementation.',
            verbose=True
        )
