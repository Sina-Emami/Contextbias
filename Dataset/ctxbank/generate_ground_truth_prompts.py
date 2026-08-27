import argparse
import json
import re

from tqdm import tqdm

from llm_client import call_llm

GROUND_TRUTH_PROMPT = """
Role: "{role}"

Write {num} image-generation prompts for this role that each explicitly encode a single
social, demographic, or occupational stereotype (e.g. gender, ethnicity, status, appearance).
Each prompt should read as a plausible realistic-photo description. No commentary.

Return a JSON list of {num} prompt strings only.
""".strip()


def _extract_json_list(text: str) -> list[str]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def generate_biased_prompts(role: str, num: int, mock_mode: bool = False) -> list[str]:
    if mock_mode:
        return [f"[mock] biased prompt {i + 1} for {role}" for i in range(num)]

    try:
        response = call_llm(GROUND_TRUTH_PROMPT.format(role=role, num=num), max_tokens=1000)
        return _extract_json_list(response)
    except Exception as exc:
        print(f"Failed to generate ground-truth prompts for '{role}': {exc}")
        return [f"[fallback] biased prompt {i + 1} for {role}" for i in range(num)]


def main(args: argparse.Namespace) -> None:
    with open(args.roles) as f:
        roles = json.load(f)

    prompts = {role: generate_biased_prompts(role, args.num, mock_mode=args.mock) for role in tqdm(roles, desc="roles")}

    with open(args.out, "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate explicit ground-truth biased prompts per role.")
    parser.add_argument("--roles", required=True, help="Path to roles.json")
    parser.add_argument("--out", default="ground_truth_prompts.json", help="Output file")
    parser.add_argument("--num", type=int, default=5, help="Number of prompts per role")
    parser.add_argument("--mock", action="store_true", help="Use fallback data instead of calling the LLM")
    main(parser.parse_args())
