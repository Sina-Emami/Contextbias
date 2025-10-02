import os
from crewai import Agent
from tools.vision_description_tool import describe_image_from_file_tool


def _describer_llm_name() -> str:
    return os.getenv("DESCRIBER_LLM", "gpt-5-mini")


def build_raw_image_describer_agent() -> Agent:
    """Agent dedicated to collecting raw descriptions from the vision tool."""
    return Agent(
        name="Image Description Collector",
        role=(
            "Call the configured vision tool for each image and return its output verbatim."
        ),
        goal=(
            "Capture the raw vision description without paraphrasing so downstream agents can reuse it."
        ),
        backstory=(
            "A meticulous recorder who trusts instrumentation over interpretation and never alters tool responses."
        ),
        tools=[describe_image_from_file_tool],
        llm=None,
        allow_delegation=False,
        verbose=True,
        max_iter=1,
        memory=False,
    )


def build_structured_image_describer_agent() -> Agent:
    """Agent that converts raw descriptions into the structured ImageAuditRecord schema."""
    return Agent(
        name="Image Description Schema Converter",
        role=(
            "Transform Stage 1 raw descriptions into the analytical schema used for downstream audits, strictly aligning categorical fields with the defined enumerations."
        ),
        goal=(
            "Emit an ImageAuditRecord that matches the structured schema, mirrors the enumerations exactly, and sets every unevidenced field to \"unknown\" instead of guessing."
        ),
        backstory=(
            "A schema specialist who tokenizes qualitative language into stable fields, refuses to hallucinate, and writes \"unknown\" whenever evidence is missing."
        ),
        tools=[],
        llm=_describer_llm_name(),
        allow_delegation=False,
        verbose=True,
        memory=False,
    )

