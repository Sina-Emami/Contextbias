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
        llm=_describer_llm_name(),
        allow_delegation=False,
        verbose=True,
        memory=False,
    )


def build_structured_image_describer_agent() -> Agent:
    """Agent that converts raw descriptions into the structured ImageAuditRecord schema."""
    return Agent(
        name="Image Description Structuring Analyst",
        role=(
            "Normalize a previously captured raw description into a strict JSON record for bias auditing."
        ),
        goal=(
            "Return an ImageAuditRecord with exhaustive, countable details and dense FeatureTokens."
        ),
        backstory=(
            "A disciplined visual metadata engineer who relies solely on recorded observations to map schema fields."
        ),
        tools=[],
        llm=_describer_llm_name(),
        allow_delegation=False,
        verbose=True,
        memory=False,
    )
