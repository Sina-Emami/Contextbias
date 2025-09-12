from typing import List
from crewai import Agent
from tools.replicate_expected_tool import get_expected_elements_replicate

# Replicate Model List — EXACTLY your list
replicate_models: List[str] = [
    "moonshotai/kimi-k2-instruct",  # "mistralai/mistral-7b-v0.1",
    "meta/meta-llama-3-8b-instruct",
    "deepseek-ai/deepseek-v3",
    "ibm-granite/granite-3.3-8b-instruct",  # "google-deepmind/gemma-3-4b-it:..."
    "microsoft/phi-3-mini-4k-instruct:e17386e6ae2e351f63783fa89f427fd0ed415524a7b3d8c122f6ac80ad0166b1",
]

# Agents for Each Replicate Model — same wording, tools only
replicate_agents = [
    Agent(
        role=f"{model} Visual Element Extractor",
        goal="Identify key visual elements in a prompt to guide image generation.",
        backstory=f"You use the model {model} to extract must-have elements for image prompts.",
        tools=[get_expected_elements_replicate],
        verbose=True,
    )
    for model in replicate_models
]

# Consensus Aggregation Agent — same as yours
consensus_agent = Agent(
    role="Consensus Content Aggregator",
    goal="Merge multiple model predictions and extract the common elements.",
    backstory="Expert in synthesizing diverse predictions into a unified output.",
    verbose=True,
)