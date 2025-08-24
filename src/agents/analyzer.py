import os
from crewai import Agent
from tools.bias_ingest_tools import ingest_description, finalize_summary
from llm.replicate_llm import ReplicateChatLLM

def build_bias_ingestor_agent() -> Agent:
    """Memory-enabled agent that ingests per-file descriptions and finalizes a summary."""
    llm = os.getenv("SUMMARY_LLM", "gpt-4o-mini")
    return Agent(
        name="Bias Ingestor",
        role="Read description JSON files and aggregate repetitions using tools.",
        goal="Ingest each file sequentially, then finalize a repetition summary.",
        backstory="Determinstic aggregator; uses tools; relies on memory for sequential flow.",
        tools=[ingest_description, finalize_summary],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=True,  # ← as requested
    )

def build_bias_reasoner_agent() -> Agent:
    """Reasoning agent (Replicate OSS 20B) that turns the summary into a BiasReport."""
    model = os.getenv("BIAS_REPLICATE_MODEL", "openai/gpt-oss-20b")
    temperature = float(os.getenv("BIAS_REPLICATE_TEMPERATURE", "0"))
    llm = ReplicateChatLLM(model=model, temperature=temperature)
    return Agent(
        name="Bias Reasoner",
        role="Infer potential dataset biases based on a repetition summary.",
        goal="Return a structured BiasReport JSON with findings and mitigations.",
        backstory="Fairness auditor focusing on representation, stereotyping, and framing.",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )
