from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from .config.agents import agents_config  # auto‑loaded dict of agents.yaml
from .config.tasks import tasks_config    # auto‑loaded dict of tasks.yaml
from .tools.custom_tools import generate_image  # ensure tool is registered

@CrewBase
class VisualBiasCrew:
    """Crew that generates, describes, analyzes bias, and filters results."""
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    @agent
    def image_generator(self) -> Agent:
        return Agent(config=self.agents_config["image_generator"])

    @agent
    def image_describer(self) -> Agent:
        return Agent(config=self.agents_config["image_describer"])

    @agent
    def must_have_gpt4omini(self) -> Agent:
        return Agent(config=self.agents_config["must_have_gpt4omini"])

    @agent
    def must_have_gpt4nano(self) -> Agent:
        return Agent(config=self.agents_config["must_have_gpt4nano"])

    @agent
    def bias_detector(self) -> Agent:
        return Agent(config=self.agents_config["bias_detector"])

    @agent
    def filter_agent(self) -> Agent:
        return Agent(config=self.agents_config["filter_agent"])

    @task
    def generate_image_task(self) -> Task:
        return Task(config=self.tasks_config["generate_image_task"])

    @task
    def describe_image_task(self) -> Task:
        return Task(config=self.tasks_config["describe_image_task"])

    @task
    def must_have_gpt4omini_task(self) -> Task:
        return Task(config=self.tasks_config["must_have_gpt4omini_task"])

    @task
    def must_have_gpt4nano_task(self) -> Task:
        return Task(config=self.tasks_config["must_have_gpt4nano_task"])

    @task
    def bias_analysis_task(self) -> Task:
        return Task(config=self.tasks_config["bias_analysis_task"])

    @task
    def filter_task(self) -> Task:
        return Task(config=self.tasks_config["filter_task"])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.image_generator(),
                self.image_describer(),
                self.must_have_gpt4omini(),
                self.must_have_gpt4nano(),
                self.bias_detector(),
                self.filter_agent(),
            ],
            tasks=[
                self.generate_image_task(),
                self.describe_image_task(),
                self.must_have_gpt4omini_task(),
                self.must_have_gpt4nano_task(),
                self.bias_analysis_task(),
                self.filter_task(),
            ],
            process=Process.sequential
        )
