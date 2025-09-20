from crewai import Crew, Process
from agents.generator import build_image_generator_agent
from tasks.generate_images import build_generate_image_task
from agents.describer import (
    build_raw_image_describer_agent,
    build_structured_image_describer_agent,
)
from tasks.describe_images import (
    build_capture_raw_description_task,
    build_structure_image_description_task,
)
from agents.analyzer import build_bias_ingestor_agent, build_bias_reasoner_agent
from tasks.analyze_bias import make_ingest_tasks, build_reason_over_summary_task
from agents.question_writer import build_question_writer_agent
from tasks.write_questions import build_write_questions_task
from agents.researcher import build_fact_checker_agent
from tasks.fact_check import build_fact_check_task
from agents.consensus import replicate_agents, consensus_agent
from tasks.consensus import prediction_tasks, consensus_task


def build_generation_crew() -> Crew:
    agent = build_image_generator_agent()
    task = build_generate_image_task(agent)
    return Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
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


def build_bias_ingest_crew(description_files, state_path, out_path) -> Crew:
    """Build a crew with N ingest tasks + 1 finalize task; agent memory is enabled."""
    agent = build_bias_ingestor_agent()
    tasks = make_ingest_tasks(agent, description_files, state_path, out_path)
    return Crew(agents=[agent], tasks=tasks, process=Process.sequential, verbose=True, memory=True)


def build_bias_reasoning_crew() -> Crew:
    agent = build_bias_reasoner_agent()
    task = build_reason_over_summary_task(agent)
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)


def build_question_writer_crew():
    agent = build_question_writer_agent()
    task = build_write_questions_task(agent)
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)


def build_fact_checker_crew() -> Crew:
    agent = build_fact_checker_agent()
    task = build_fact_check_task(agent)
    return Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)


def build_consensus_crew() -> Crew:
    return Crew(
        agents=replicate_agents + [consensus_agent],
        tasks=prediction_tasks + [consensus_task],
        process=Process.sequential,
        verbose=True,
    )

