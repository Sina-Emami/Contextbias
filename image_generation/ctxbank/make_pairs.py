# ctxbank/make_pairs.py
import json
import argparse
import itertools
from random import seed, sample

PAIR_TEMPLATE = "a photo of a {ROLE} {VERB} {CONTEXT}"

def build_minimal_pairs(context_bank, seeds_per_pair=10, random_seed=42):
    seed(random_seed)
    pairs = []
    for role, axes in context_bank.items():
        # Get ACTION and LOCATION lists
        actions_rel = [it["item"] for it in axes.get("ACTION", {}).get("related", [])]
        actions_unr = [it["item"] for it in axes.get("ACTION", {}).get("unrelated", [])]
        locations_rel = [it["item"] for it in axes.get("LOCATION", {}).get("related", [])]
        locations_unr = [it["item"] for it in axes.get("LOCATION", {}).get("unrelated", [])]

        # Orthogonal CA-R pairs (related action × related location)
        for a in actions_rel:
            for l in locations_rel:
                seeds = sample(range(1_000_000), seeds_per_pair)
                pairs.append({
                    "role": role,
                    "axis": "ACTION_LOCATION",
                    "type": "CA-R",
                    "action": a,
                    "location": l,
                    "seeds": seeds
                })

        # Orthogonal CA-U pairs (unrelated action × unrelated location)
        for a in actions_unr:
            for l in locations_unr:
                seeds = sample(range(1_000_000), seeds_per_pair)
                pairs.append({
                    "role": role,
                    "axis": "ACTION_LOCATION",
                    "type": "CA-U",
                    "action": a,
                    "location": l,
                    "seeds": seeds
                })

        # Minimal pairs: CA-R vs CA-R, CA-U vs CA-U (same location, different action)
        for l in locations_rel:
            for a1, a2 in itertools.combinations(actions_rel, 2):
                seeds = sample(range(1_000_000), seeds_per_pair)
                pairs.append({
                    "role": role,
                    "axis": "ACTION_LOCATION",
                    "type": "CA-R-PAIR",
                    "action_left": a1,
                    "action_right": a2,
                    "location": l,
                    "seeds": seeds
                })
        for l in locations_unr:
            for a1, a2 in itertools.combinations(actions_unr, 2):
                seeds = sample(range(1_000_000), seeds_per_pair)
                pairs.append({
                    "role": role,
                    "axis": "ACTION_LOCATION",
                    "type": "CA-U-PAIR",
                    "action_left": a1,
                    "action_right": a2,
                    "location": l,
                    "seeds": seeds
                })

        # CF (context-free) prompts
        seeds = sample(range(1_000_000), seeds_per_pair)
        pairs.append({
            "role": role,
            "axis": "CF",
            "type": "CF",
            "seeds": seeds
        })
    return pairs

def main(args):
    with open(args.context_bank, "r") as f:
        bank = json.load(f)
    pairs = build_minimal_pairs(bank, seeds_per_pair=args.seeds)
    with open(args.out, "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_bank", required=True)
    parser.add_argument("--out", default="pairs.json")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    main(args)