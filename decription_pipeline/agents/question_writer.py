import os
from crewai import Agent, LLM
from llm.replicate_llm import ReplicateChatLLM

def build_question_writer_agent() -> Agent:
    # todo must check what the problem with using oss 20b models
    # Use CrewAI/LiteLLM route for Replicate OSS-20B (requires REPLICATE_API_KEY)
    # model = os.getenv("BIAS_REPLICATE_MODEL", "replicate/openai/gpt-oss-20b")
    # temperature = float(os.getenv("BIAS_REPLICATE_TEMPERATURE", "0.0"))
    # question_llm = LLM(model="replicate/openai/gpt-oss-20b", temperature=0.0)
    # question_llm = ReplicateChatLLM(model=model, temperature=temperature)

    return Agent(
        role="Bias Question Generator",
        goal="Turn only verifiable visual-pattern signals into precise, globally scoped statistical questions with atomic answers.",
        backstory=(
            "You analyze inputs describing visual patterns and potential biases in any domain. "
            "Output ONLY questions that can be answered by a SINGLE percentage (0–100%), a ONE-WORD category, or YES/NO. "
            "Scope must be GLOBAL (worldwide/across countries). "
            "Focus on attributes commonly measured across countries (e.g., sex/gender shares, age cohorts, race/ethnicity as commonly reported, "
            "occupational/workforce composition, sector/institution presence, geographic presence). "
            "EXCLUDE non-measurable or subjective items (emotions, fashion unless a job uniform, composition/camera/angles/colors/symbols). "
            "DE-DUPLICATE: for binary attributes, ask only one side; for multi-class, ask separate single-category percentage questions. "
            "Each question must be fully standalone."
        ),
        llm="gpt-5-mini", #question_llm,
        allow_delegation=False,
        verbose=True,
    )