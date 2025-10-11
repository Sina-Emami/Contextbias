import os
from crewai import Agent

from tools.vision_description_tool import describe_image_from_file_tool


def _describer_llm_name() -> str:
    return os.getenv("DESCRIBER_LLM", "gpt-5-nano")


def build_image_describer_agent() -> Agent:
    """Create a single agent that captures the vision tool output and emits schema-ready JSON."""
    return Agent(
        name="Image Schema Describer",
        role=(
            "Call the configured vision tool for each image and return its output verbatim. Then translate the "
            "tool's authoritative narrative into the ImageAuditRecord schema without omitting required fields."
        ),
        goal=(
            "Produce a strictly valid ImageAuditRecord JSON payload that mirrors the tool observations, "
            "uses only declared enum tokens, and writes 'unknown' or [] whenever evidence is missing."
        ),
        backstory=(
            "A forensic cataloger who trusts instrument readings above intuition. They read the vision tool "
            "output carefully, organize it into structured cohorts, and never invent facts beyond what the "
            "tool reports."
        ),
        tools=[describe_image_from_file_tool],
        llm=None,  # _describer_llm_name(),
        allow_delegation=False,
        verbose=True,
        memory=False,
        max_iter=1,
    )


__all__ = ["build_image_describer_agent"]
