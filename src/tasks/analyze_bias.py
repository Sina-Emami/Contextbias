from pathlib import Path
from typing import List
from crewai import Task
from schemas.bias import BiasReport

def make_ingest_tasks(agent, files: List[Path], state_path: Path, out_path: Path) -> list[Task]:
    """Create one ingest task per file (so the agent processes them sequentially with memory),
    then one finalize task that returns the JSON summary."""
    tasks: list[Task] = []
    for fp in files:
        tasks.append(Task(
            description=(
                "Call the tool 'ingest_description' on this file and state:\n"
                f"- description_path: {fp.as_posix()}\n"
                f"- state_path: {state_path.as_posix()}\n"
                "Return 'ok'."
            ),
            expected_output="ok",
            agent=agent,
        ))
    tasks.append(Task(
        description=(
            "Now call 'finalize_summary' to recompute totals and write the JSON file:\n"
            f"- state_path: {state_path.as_posix()}\n"
            f"- out_path: {out_path.as_posix()}\n"
            "Return ONLY the JSON summary."
        ),
        expected_output="A JSON dictionary with repetition summary.",
        agent=agent,
    ))
    return tasks

# You can paste a longer guide if you like; kept minimal per your request
BIAS_REASONING_GUIDE = (
    "Analyze the repetition summary and infer potential biases. "
    "Return ONLY valid JSON for BiasReport.\n\n"
    "SUMMARY_JSON:\n{summary_json}\n"
    "EXTRA_CONTEXT:\n{extra_context}"
)

def build_reason_over_summary_task(agent) -> Task:
    return Task(
        description=BIAS_REASONING_GUIDE,
        agent=agent,
        expected_output="A valid JSON object matching BiasReport.",
        output_pydantic=BiasReport,
    )
