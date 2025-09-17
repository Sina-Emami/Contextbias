# ctxbank/generate_candidates.py
import json
import argparse
from tqdm import tqdm
from image_generation.ctxbank.llm_client import call_llm

AXES = ["ACTION", "LOCATION"]

PASS_A_PROMPT = """
You are given a ROLE: "{role}".
Produce JSON with fields for each axis: {axes_list}.
For each axis produce two lists: "related" and "unrelated".
Each list should be an array of objects: {{ "item": "...", "rationale": "one-line" }}.
Constraints:
- Items must be short, concrete, parallel (nouns/gerunds), no demographic terms, no brands, no names.
- Related items should be typical to the role. Unrelated items should be plausible photos but orthogonal.
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