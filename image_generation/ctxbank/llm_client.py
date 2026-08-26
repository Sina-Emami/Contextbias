# ctxbank/llm_client.py


import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Default to cheapest model

if not OPENAI_API_KEY:
    print("Warning: OpenAI API key not found in .env file")
    print("Required variable: OPENAI_API_KEY")
    print("Please update the .env file with your actual OpenAI API key")
    raise RuntimeError("Set OPENAI_API_KEY in .env")

client = OpenAI(
    api_key=OPENAI_API_KEY
)

def call_llm(prompt: str, max_tokens=30000):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a JSON generator. Always output only valid JSON, no explanations."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,  # Standard OpenAI API uses max_tokens
        model=OPENAI_MODEL
    )

    content = response.choices[0].message.content
    print(content)
    if not content or content.strip() == "":
        raise ValueError("LLM returned empty content.")
    return content

