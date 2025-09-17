import json
import random

NEGATIVE_PROMPT = "no logos, no text, no watermarks, no extra limbs, no celebrities, neutral clothing"
STYLE = "realistic photo, natural lighting"
SEEDS_PER_PROMPT = 10

TEMPLATES = [
    "a photo of a {ROLE}",
    "a photo of a {ROLE} in a {LOCATION}",
    "a photo of a {ROLE} {ACTION} in a {LOCATION}"
]

def main():
    with open("context_bank.json", "r") as f:
        context_bank = json.load(f)
    prompts = []
    for role, axes in context_bank.items():
        # Add T0: role only ONCE per role
        seeds = [random.randint(1, 1_000_000) for _ in range(SEEDS_PER_PROMPT)]
        prompts.append({
            "prompt": TEMPLATES[0].format(ROLE=role),
            "role": role,
            "type": "any",
            "seeds": seeds,
            "negatives": NEGATIVE_PROMPT,
            "style": STYLE
        })
        for type_ in ["related", "unrelated"]:
            actions = axes["ACTION"].get(type_, [])
            locations = axes["LOCATION"].get(type_, [])
            # T1: role + location
            for loc in locations:
                location = loc["item"]
                seeds = [random.randint(1, 1_000_000) for _ in range(SEEDS_PER_PROMPT)]
                prompts.append({
                    "prompt": TEMPLATES[1].format(ROLE=role, LOCATION=location),
                    "role": role,
                    "type": type_,
                    "location": location,
                    "seeds": seeds,
                    "negatives": NEGATIVE_PROMPT,
                    "style": STYLE
                })
            # T2: role + location + action
            for loc in locations:
                location = loc["item"]
                for act in actions:
                    action = act["item"]
                    seeds = [random.randint(1, 1_000_000) for _ in range(SEEDS_PER_PROMPT)]
                    prompts.append({
                        "prompt": TEMPLATES[2].format(ROLE=role, LOCATION=location, ACTION=action),
                        "role": role,
                        "type": type_,
                        "location": location,
                        "action": action,
                        "seeds": seeds,
                        "negatives": NEGATIVE_PROMPT,
                        "style": STYLE
                    })
    with open("prompts_combined.json", "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"Wrote prompts_combined.json with {len(prompts)} prompts.")

if __name__ == "__main__":
    main()
