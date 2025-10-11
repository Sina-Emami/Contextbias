from crewai import Crew, Process
from agents.describer import build_image_describer_agent
from agents.summary_reporter import build_summary_report_agent
from tasks.describe_images import build_describe_image_task
from tasks.summary_report import build_summary_report_task


def build_image_description_crew() -> Crew:
    agent = build_image_describer_agent()
    task = build_describe_image_task(agent)
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
