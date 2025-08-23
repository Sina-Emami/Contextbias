from crewai import Task
from ..schemas.scenario import ImageGenerationOutput


def build_generate_image_task(agent) -> Task:
    return Task(
        description=(
            "Use the `generate_image` tool to generate an image for the given prompt:\n\n{prompt}\n\n "
            "Return only a JSON object matching this schema:\n"
            "{\n"
            '  "id": "<use the id that tool return to the agent>",\n'
            '  "image_url": "<public URL of the generated image that is in the tool return>",\n'
            '  "prompt_used": "<the exact prompt string used for generation>"\n'
            "}\n\n"
            "The `id` must be a unique string (e.g., UUID or image_1, image_2)."
        ),
        expected_output=(
            'A JSON object with keys "id", "image_url" (string URL) and "prompt_used" (string).'
        ),
        agent=agent,
        output_pydantic=ImageGenerationOutput,
    )