from crewai import Crew, Process
from agents import ScientificAgents
from tasks import ScientificTasks

class ScientificResearchCrew:
    def __init__(self, topic: str):
        self.topic = topic
        self.agents = ScientificAgents()
        self.tasks = ScientificTasks()

    def run(self):
        # 1. Instantiate Agents
        investigator = self.agents.chief_investigator()
        architect = self.agents.methodology_architect()
        critic = self.agents.scientific_critic()
        engineer = self.agents.experimental_engineer()
        
        # 2. Instantiate Tasks
        t1_review = self.tasks.literature_review_task(investigator, self.topic)
        
        t2_decompose = self.tasks.decomposition_task(architect, [t1_review])
        
        t3_critique = self.tasks.negative_analysis_task(critic, [t2_decompose])
        
        t4_experiment = self.tasks.experimentation_task(engineer, [t2_decompose, t3_critique])
        
        t5_publish = self.tasks.publication_task(investigator, [t1_review, t2_decompose, t3_critique, t4_experiment])

        # 3. Form Crew
        crew = Crew(
            agents=[investigator, architect, critic, engineer],
            tasks=[t1_review, t2_decompose, t3_critique, t4_experiment, t5_publish],
            process=Process.sequential, # Pipeline: Review -> Decompose -> Critique -> Experiment -> Publish
            verbose=True
        )

        return crew.kickoff()
