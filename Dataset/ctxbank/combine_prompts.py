import argparse
import json
import random

NEGATIVE_PROMPT = "no logos, no text, no watermarks, no extra limbs, no celebrities, neutral clothing"
STYLE = "realistic photo"
SEEDS_PER_PROMPT = 20

TEMPLATE_ROLE_ONLY = "a photo of a {role}"
TEMPLATE_ROLE_LOCATION = "a photo of a {role} in a {location}"
TEMPLATE_ROLE_LOCATION_ACTION = "a photo of a {role} {action} in a {location}"


def _seeds() -> list[int]:
    return [random.randint(1, 1_000_000) for _ in range(SEEDS_PER_PROMPT)]


def _base_entry(prompt: str, role: str, prompt_type: str, **extra) -> dict:
    return {
        "prompt": prompt,
        "role": role,
        "type": prompt_type,
        "seeds": _seeds(),
        "negatives": NEGATIVE_PROMPT,
        "style": STYLE,
        **extra,
    }


def prompts_for_bucket(role: str, prompt_type: str, pairs: list[dict]) -> list[dict]:
    if not pairs:
        return []

    entries = [_base_entry(TEMPLATE_ROLE_ONLY.format(role=role), role, prompt_type)]

    for pair in pairs:
        location = pair["location"]
        entries.append(
            _base_entry(
                TEMPLATE_ROLE_LOCATION.format(role=role, location=location),
                role,
                prompt_type,
                location=location,
            )
        )

    for pair in pairs:
        location, action = pair["location"], pair["action"]
        entries.append(
            _base_entry(
                TEMPLATE_ROLE_LOCATION_ACTION.format(role=role, location=location, action=action),
                role,
                prompt_type,
                location=location,
                action=action,
            )
        )

    return entries


def build_prompts(context_bank: dict) -> list[dict]:
    prompts = []
    for role, buckets in context_bank.items():
        for prompt_type in ("related", "unrelated"):
            prompts.extend(prompts_for_bucket(role, prompt_type, buckets.get(prompt_type, [])))
    return prompts


def dedupe_by_prompt_text(prompts: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for entry in prompts:
        seen.setdefault(entry["prompt"], entry)
    return list(seen.values())


def main(args: argparse.Namespace) -> None:
    with open(args.context_bank) as f:
        context_bank = json.load(f)

    prompts = dedupe_by_prompt_text(build_prompts(context_bank))

    with open(args.out, "w") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out} with {len(prompts)} unique prompts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand a context bank into image-generation prompts.")
    parser.add_argument("--context_bank", default="context_bank.json", help="Input context bank file")
    parser.add_argument("--out", default="prompts_combined.json", help="Output prompts file")
    main(parser.parse_args())
