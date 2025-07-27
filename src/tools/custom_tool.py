from openai import OpenAI
from crewai.tools import tool

client = OpenAI()

@tool("generate_image")
def generate_image(prompt: str) -> str:
    """
    Uses OpenAI’s DALL·E‑3 to generate one image and returns its URL.
    """
    resp = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        n=1,
        size="512x512",
    )
    return resp.data[0].url
