from crewai import Crew, Process
from agents.summary_reporter import build_summary_report_agent
from tasks.summary_report import build_summary_report_task
from agents.describer import (
    build_raw_image_describer_agent,
    build_structured_image_describer_agent,
)
from tasks.describe_images import (
    build_capture_raw_description_task,
    build_structure_image_description_task,
)


def build_raw_description_crew() -> Crew:
    agent = build_raw_image_describer_agent()
    task = build_capture_raw_description_task(agent)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def build_structured_description_crew() -> Crew:
    agent = build_structured_image_describer_agent()
    task = build_structure_image_description_task(agent)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )


def build_summary_report_crew(counts_path) -> Crew:
    agent = build_summary_report_agent()
    task = build_summary_report_task(agent, counts_path)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )
