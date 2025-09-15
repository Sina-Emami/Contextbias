# ctxbank/llm_client.py


import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("Set AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT in .env")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def call_llm(prompt: str, max_tokens=30000):
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a JSON generator. Always output only valid JSON, no explanations."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=max_tokens,   # Azure variant requires this
        model=AZURE_OPENAI_DEPLOYMENT
    )

    content = response.choices[0].message.content
    print(content)
    if not content or content.strip() == "":
        raise ValueError("LLM returned empty content.")
    return content

