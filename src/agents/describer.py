
import os
from crewai import Agent
from tools.vision_description_tool import (
    describe_image_from_file_tool,
    describe_image_from_url_tool,
)


def build_image_describer_agent() -> Agent:
    llm = os.getenv("DESCRIBER_LLM", "gpt-5-mini")  # default changed per request
    return Agent(
        name="Image Description Structuring Analyst",
        role=(
            "Normalize free-form image descriptions into a strict JSON record for bias auditing. "
            "If no raw description is provided, call a vision description tool first."
        ),
        goal="Return an ImageAuditRecord with exhaustive, countable details and dense FeatureTokens.",
        backstory=(
            "Meticulous visual metadata engineer. Uses evidence-only for sensitive traits; prefers 'unknown' over guessing."
        ),
        tools=[describe_image_from_file_tool, describe_image_from_url_tool],
        llm=llm,
        allow_delegation=False,
        verbose=True,
        memory=False,
    )