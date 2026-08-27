import argparse
import json
import re

from tqdm import tqdm

from llm_client import call_llm

CANDIDATE_PROMPT = """
Role: "{role}"

Produce two lists of action-location pairs for this role: "related" and "unrelated".
Each list item is an object with "action", "location", and "rationale" fields.

- "related" pairs describe an action and location that are typical for someone in this role.
- "unrelated" pairs describe a plausible, realistic everyday action-location combination that
  has nothing to do with the role.
- Keep items short and concrete (nouns/gerunds). No demographic terms, brand names, or
  person names.

Return JSON only, shaped as:
{{"related": [{{"action": "...", "location": "...", "rationale": "..."}}, ...],
  "unrelated": [{{"action": "...", "location": "...", "rationale": "..."}}, ...]}}
""".strip()


def _extract_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _fallback_candidates(role: str) -> dict:
    return {
        "related": [
            {
                "action": f"working as {role}",
                "location": "workplace",
                "rationale": f"Typical work activity for a {role}.",
            }
        ],
        "unrelated": [
            {
                "action": "reading a book",
                "location": "home",
                "rationale": "Everyday activity unrelated to the role.",
            }
        ],
    }


def generate_for_role(role: str, mock_mode: bool = False) -> dict:
    if mock_mode:
        return _fallback_candidates(role)

    try:
        response = call_llm(CANDIDATE_PROMPT.format(role=role), max_tokens=1000)
        return _extract_json_object(response)
    except Exception as exc:
        print(f"Failed to generate candidates for '{role}': {exc}")
        return _fallback_candidates(role)


def main(args: argparse.Namespace) -> None:
    with open(args.roles) as f:
        roles = json.load(f)

    candidates = {role: generate_for_role(role, mock_mode=args.mock) for role in tqdm(roles, desc="roles")}

    with open(args.out, "w") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out} ({len(candidates)} roles)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate related/unrelated action-location candidates per role.")
    parser.add_argument("--roles", required=True, help="Path to roles.json")
    parser.add_argument("--out", default="candidates.json", help="Output file")
    parser.add_argument("--mock", action="store_true", help="Use fallback data instead of calling the LLM")
    main(parser.parse_args())
