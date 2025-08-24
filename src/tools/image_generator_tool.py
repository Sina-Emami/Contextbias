import os
import uuid
from typing import Dict, Any
from crewai.tools import tool
from openai import OpenAI

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")


@tool("generate_image")
def generate_image(prompt: str) -> Dict[str, Any]:
    """Generate ONE image and return a dict: {id, image_url, prompt_used}."""
    if not prompt or not prompt.strip():
        return {"error": "Empty prompt provided for image generation."}
    try:
        resp = _client.images.generate(
            model=_IMAGE_MODEL,
            prompt=prompt.strip(),
            n=1,
            size="1024x1024",
        )
        image_url = resp.data[0].url  # If your org returns b64, tell me and I'll decode & save.
        return {
            "id": f"image_{uuid.uuid4().hex[:8]}",
            "image_url": image_url,
            "prompt_used": prompt.strip(),
        }
    except Exception as e:
        return {"error": f"Image generation failed: {str(e)}"}