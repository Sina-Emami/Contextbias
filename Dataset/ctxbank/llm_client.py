import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file before calling the LLM."
        )
    return OpenAI(api_key=api_key)


def call_llm(prompt: str, max_tokens: int = 30000, model: str | None = None) -> str:
    client = _get_client()
    resolved_model = model or os.getenv("OPENAI_MODEL", DEFAULT_MODEL)

    response = client.chat.completions.create(
        model=resolved_model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "system",
                "content": "You are a JSON generator. Always output only valid JSON, no explanations.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("LLM returned an empty response.")
    return content
