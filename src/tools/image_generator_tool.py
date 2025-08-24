import os
from uuid import uuid4
from typing import Any
from crewai.tools import tool
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_IMAGE_MODEL = os.getenv("IMAGE_MODEL", "dall-e-2")


@tool("generate_image")
def generate_image(prompt: str) -> dict[str, Any]:
    """
    Generate a single 1024x1024 image from a natural-language prompt using the
    configured OpenAI Images API model.

    Args:
        prompt (str): The scenario or description to render. Must be non-empty.

    Returns:
        Dict[str, Any]: On success, a dict with:
            {
              "id": "<unique image id>",
              "image_url": "<public URL of the generated image>",
              "prompt_used": "<the exact prompt string used>"
            }
        On failure, a dict with:
            { "error": "<error message>" }

    Notes:
        - The model is taken from the IMAGE_MODEL env var (default: "gpt-image-1").
        - This implementation expects a URL at resp.data[0].url. If your org returns
          base64 (e.g., `b64_json`) instead of a URL, add a decode-and-save branch.
    """
    if not prompt or not prompt.strip():
        return {"error": "Empty prompt provided for image generation."}
    try:
        resp = _client.images.generate(
            model="dall-e-2", #_IMAGE_MODEL,
            prompt=prompt.strip(),
            n=1,
            size="1024x1024",
        )
        image_url = resp.data[0].url  # If your org returns b64, tell me and I'll decode & save.
        return {
            "id": f"image_{uuid4().hex[:8]}",
            "image_url": image_url,
            "prompt_used": prompt.strip(),
        }
    except Exception as e:
        return {"error": f"Image generation failed: {str(e)}"}