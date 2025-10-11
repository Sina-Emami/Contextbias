from crewai import Crew, Process
from agents.describer import build_image_describer_agent
from tasks.describe_images import build_describe_image_task


def build_image_description_crew() -> Crew:
    agent = build_image_describer_agent()
    task = build_describe_image_task(agent)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )