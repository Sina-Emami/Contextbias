import os
from crewai import Agent
from tools.image_generator_tool import generate_image


def build_image_generator_agent() -> Agent:
    # agent_llm = os.getenv("AGENT_LLM", "gpt-4o-mini")
    return Agent(
        role="Visionary Image Generator",
        goal=(
            "Generate a compelling and realistic image from the provided scenario using the generate_image tool."
        ),
        backstory=(
            "Creative AI skilled at visualizing scenes from text. Always use the tool generate_image for image generation."
        ),
        llm= None, # agent_llm,
        tools=[generate_image],
        verbose=True,
        allow_delegation=False,
    )