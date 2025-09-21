# ctxbank/generate_candidates.py
import json
import argparse
from tqdm import tqdm
from image_generation.ctxbank.llm_client import call_llm

AXES = ["ACTION", "LOCATION"]

PASS_A_PROMPT = """
You are given a ROLE: "{role}".
Produce JSON with two top-level keys: "related" and "unrelated".
Each should be a list of objects, each object containing both "action" and "location" fields, and a "rationale" field. For "related", the action and location should both be typical for the role. For "unrelated", the action-location pair should be a plausible everyday scenario, but not related to the role.
Constraints:
- Items must be short, concrete, parallel (nouns/gerunds), no demographic terms, no brands, no names.
- Related items should be typical to the role.
- Unrelated items should be plausible everyday action-location pairs, not related to the role, but realistic for a photo.
- Output valid JSON only (no extra commentary).
Return one candidates per list.
""".strip()

def generate_for_role(role, axes=AXES):
    prompt = PASS_A_PROMPT.format(role=role, axes_list=", ".join(axes))
    resp = call_llm(prompt, max_tokens=30000)
    print(f"LLM response for role '{role}':\n{resp}\n")  # Add this line
    # The model must return JSON. We attempt to parse:
    try:
        data = json.loads(resp)
    except Exception as e:
        # If the model returns text around JSON, try to recover substring
        start = resp.find("{")
        end = resp.rfind("}") + 1
        data = json.loads(resp[start:end])
    return data

def main(args):
    with open(args.roles, "r") as f:
        roles = json.load(f)
    out = {}
    for role in tqdm(roles, desc="roles"):
        out[role] = generate_for_role(role)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", required=True)
    parser.add_argument("--out", default="candidates.json")
    args = parser.parse_args()
    main(args)