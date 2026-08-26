# generate_ground_truth_prompts.py
import json
import argparse
from tqdm import tqdm
from llm_client import call_llm

BIAS_PROMPT_TEMPLATE = """
You are given a ROLE: "{role}".
Generate {num} image generation prompts that explicitly include social, demographic, or occupational biases related to this role. Each prompt should be a realistic photo description that would visually express a stereotype or bias about the role. Do NOT include any commentary or explanation—just output a JSON list of prompt strings.
Constraints:
- Prompts should be short, concrete, and plausible for image generation.
- Each prompt must clearly encode a bias (e.g., gender, ethnicity, status, appearance, etc.)
- Output valid JSON only (no extra commentary).
""".strip()

def generate_biased_prompts(role, num, mock_mode=False):
    if mock_mode:
        # Return mock biased prompts for testing
        return [f"Mock biased prompt {i+1} for {role}" for i in range(num)]
    
    prompt = BIAS_PROMPT_TEMPLATE.format(role=role, num=num)
    try:
        resp = call_llm(prompt, max_tokens=1000)
        try:
            data = json.loads(resp)
        except Exception:
            start = resp.find("[")
            end = resp.rfind("]") + 1
            data = json.loads(resp[start:end])
        return data
    except Exception as e:
        print(f"Error calling LLM for role '{role}': {e}")
        # Return default prompts on API failure
        return [f"Default biased prompt {i+1} for {role}" for i in range(num)]

def main(args):
    with open(args.roles, "r") as f:
        roles = json.load(f)
    out = {}
    for role in tqdm(roles, desc="roles"):
        out[role] = generate_biased_prompts(role, args.num, mock_mode=args.mock)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out} with ground truth biased prompts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ground truth biased prompts for each role.")
    parser.add_argument("--roles", type=str, required=True, help="Path to roles.json")
    parser.add_argument("--out", type=str, default="ground_truth_prompts.json", help="Output file")
    parser.add_argument("--num", type=int, default=5, help="Number of biased prompts per role")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of calling LLM API")
    args = parser.parse_args()
    main(args)
