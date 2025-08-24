from crewai import Crew, Process
from agents.generator import build_image_generator_agent
from tasks.generate_images import build_generate_image_task
from agents.describer import build_image_describer_agent
from tasks.describe_images import build_describe_image_task
from agents.analyzer import build_bias_ingestor_agent, build_bias_reasoner_agent
from tasks.analyze_bias import make_ingest_tasks, build_reason_over_summary_task

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

def build_bias_ingest_crew(description_files, state_path, out_path) -> Crew:
    """Build a crew with N ingest tasks + 1 finalize task; agent memory is enabled."""
    agent = build_bias_ingestor_agent()
    tasks = make_ingest_tasks(agent, description_files, state_path, out_path)
    return Crew(agents=[agent], tasks=tasks, process=Process.sequential, verbose=True, memory=True)

def build_bias_reasoning_crew() -> Crew:
    agent = build_bias_reasoner_agent()
    task = build_reason_over_summary_task(agent)
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
