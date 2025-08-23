from crewai import Crew, Process
from .agents.generator import build_image_generator_agent
from .tasks.generate_images import build_generate_image_task
from .agents.describer import build_image_describer_agent
from .tasks.describe_images import build_describe_image_task


def build_generation_crew() -> Crew:
    agent = build_image_generator_agent()
    task = build_generate_image_task(agent)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def build_description_crew() -> Crew:
    agent = build_image_describer_agent()
    task = build_describe_image_task(agent)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
