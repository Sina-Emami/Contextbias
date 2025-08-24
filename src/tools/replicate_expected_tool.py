from typing import List
from crewai.tools import tool
import replicate
import json

@tool("get_expected_elements_replicate")
def get_expected_elements_replicate(model: str, prompt: str) -> List[str]:
    """Calls a Replicate model and returns a list of key elements for the image."""
    output = replicate.run(
        model,
        input={
            "prompt": (
                f"Given this image generation prompt: '{prompt}', "
                "list the key objects and details that must appear in the image. "
                "Respond with only a JSON list of strings like [\"object1\", \"object2\", ...]"
            )
        }
    )

    if isinstance(output, list):
        return output
    elif isinstance(output, str):
        try:
            return json.loads(output)
        except:
            content = output.strip("[]")
            return [item.strip().strip('"') for item in content.split(",")]
    else:
        return []