from crewai import Task

class ScientificTasks:
    def literature_review_task(self, agent, topic):
        return Task(
            description=f"Conduct a comprehensive literature review on '{topic}'. Use Tavily to find at least 5 relevant papers/repos. Read local context if available.",
            expected_output="A summary report of current state-of-the-art methods and identify gaps.",
            agent=agent
        )

    def decomposition_task(self, agent, context):
        return Task(
            description="Decompose the research problem into 'Smallest Functional Blocks'. Create a 'Paper2Code' architectural plan.",
            expected_output="A detailed list of functional blocks (Data, Model, Train, Eval) with interfaces defined.",
            agent=agent,
            context=context # Context from review
        )

    def negative_analysis_task(self, agent, context):
        return Task(
            description="Critique the proposed architectural plan. Perform 'Negative Analysis'. Identify why it might fail.",
            expected_output="A critique report listing at least 3 potential failure modes or logical flaws.",
            agent=agent,
            context=context # Context from plan
        )

    def experimentation_task(self, agent, context):
        return Task(
            description="Write Python code to implement the refined plan. Simulate execution and report results.",
            expected_output="Executable Python code blocks and a summary of experimental results (simulated).",
            agent=agent,
            context=context # Context from plan + critique
        )

    def publication_task(self, agent, context):
        return Task(
            description="Synthesize all findings into a final research paper. Include Abstract, Methodology, Results, and Negative Analysis.",
            expected_output="A markdown formatted research paper.",
            agent=agent,
            context=context # All previous contexts
        )
