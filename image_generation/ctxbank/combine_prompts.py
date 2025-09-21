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
        # Related case
        related = axes.get("related", [])
        if related:
            type_ = "related"
            # T0: role only
            seeds = [random.randint(1, 1_000_000) for _ in range(SEEDS_PER_PROMPT)]
            prompts.append({
                "prompt": TEMPLATES[0].format(ROLE=role),
                "role": role,
                "type": type_,
                "seeds": seeds,
                "negatives": NEGATIVE_PROMPT,
                "style": STYLE
            })
            # T1: role + location
            for pair in related:
                location = pair["location"]
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
            for pair in related:
                location = pair["location"]
                action = pair["action"]
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
        # Unrelated case
        unrelated = axes.get("unrelated", [])
        if unrelated:
            type_ = "unrelated"
            # T0: role only
            seeds = [random.randint(1, 1_000_000) for _ in range(SEEDS_PER_PROMPT)]
            prompts.append({
                "prompt": TEMPLATES[0].format(ROLE=role),
                "role": role,
                "type": type_,
                "seeds": seeds,
                "negatives": NEGATIVE_PROMPT,
                "style": STYLE
            })
            # T1: role + location
            for pair in unrelated:
                location = pair["location"]
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
            for pair in unrelated:
                location = pair["location"]
                action = pair["action"]
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
    # Remove duplicate prompts (by 'prompt' string)
    unique = {}
    for entry in prompts:
        key = entry["prompt"]
        if key not in unique:
            unique[key] = entry
    deduped_prompts = list(unique.values())
    with open("prompts_combined.json", "w") as f:
        json.dump(deduped_prompts, f, indent=2, ensure_ascii=False)
    print(f"Wrote prompts_combined.json with {len(deduped_prompts)} unique prompts.")

if __name__ == "__main__":
    main()
