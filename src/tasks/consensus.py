from crewai import Task
from agents.consensus import replicate_models, replicate_agents, consensus_agent
from schemas.consensus import ConsensusOutput

# Tasks for each model — your description intact
prediction_tasks = []
for idx, agent in enumerate(replicate_agents):
    t = Task(
        description=(
            f"Using your model ({replicate_models[idx]}), analyze this image prompt:\n\n{{prompt}}\n\n"
            "Return a JSON list of required visual elements and key objects that should be in the image."
        ),
        expected_output="A JSON list of required image elements.",
        agent=agent,
    )
    prediction_tasks.append(t)

# Consensus task — SAME description + 1 extra line so it knows model names order
consensus_task = Task(
    description=(
        "You will receive outputs from several models. "
        "Identify items that appear in at least 70% of the outputs. "
        "Return a JSON matching the ConsensusOutput schema.\n\n"
        "Model names in order (align with the prior outputs): {model_names}\n"
        'The "individual_predictions" array MUST include {"model_name": "<slug>", "required_elements":[...]} for each.'
    ),
    expected_output="A JSON object matching the ConsensusOutput model.",
    agent=consensus_agent,
    output_pydantic=ConsensusOutput,   # <- correct arg for Pydantic v2
)