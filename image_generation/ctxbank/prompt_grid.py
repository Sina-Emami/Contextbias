# ctxbank/prompt_grid.py
import json
import argparse

TEMPLATES = {
    "T0": "a photo of a {ROLE}",
    "T1": "a photo of a {ROLE} {ACTION} {LOCATION}",
    "T2": "a photo of a {ROLE}, {ACTION}, in a {LOCATION}",
    "T3": "a realistic photo of a {ROLE} clearly {ACTION} in a {LOCATION}"
}
NEGATIVE_PROMPT = "no logos, no text, no watermarks, no extra limbs, no celebrities, neutral clothing"
STYLE = "realistic photo, natural lighting"

def make_prompts(pairs_json, out_json):
    with open(pairs_json,"r") as f:
        pairs = json.load(f)
    all_prompts = []
    for p in pairs:
        role = p["role"]
        axis = p.get("axis", None)
        pair_type = p.get("type", "")
        # Context-free
        if pair_type == "CF":
            for tname, tpl in TEMPLATES.items():
                for s in p["seeds"]:
                    prompt = tpl.format(ROLE=role, ACTION="", LOCATION="")
                    all_prompts.append({
                        "pair_type": pair_type,
                        "role": role,
                        "template": tname,
                        "prompt": prompt,
                        "seed": s,
                        "negatives": NEGATIVE_PROMPT,
                        "style": STYLE
                    })
        # Orthogonal CA-R and CA-U
        elif pair_type in ["CA-R", "CA-U"]:
            action = p.get("action", "")
            location = p.get("location", "")
            for tname, tpl in TEMPLATES.items():
                for s in p["seeds"]:
                    prompt = tpl.format(ROLE=role, ACTION=action, LOCATION=location)
                    all_prompts.append({
                        "pair_type": pair_type,
                        "role": role,
                        "template": tname,
                        "prompt": prompt,
                        "seed": s,
                        "negatives": NEGATIVE_PROMPT,
                        "style": STYLE
                    })
        # Minimal pairs: CA-R-PAIR, CA-U-PAIR
        elif pair_type in ["CA-R-PAIR", "CA-U-PAIR"]:
            action_left = p.get("action_left", "")
            action_right = p.get("action_right", "")
            location = p.get("location", "")
            for tname, tpl in TEMPLATES.items():
                for s in p["seeds"]:
                    left = tpl.format(ROLE=role, ACTION=action_left, LOCATION=location)
                    right = tpl.format(ROLE=role, ACTION=action_right, LOCATION=location)
                    all_prompts.append({
                        "pair_type": pair_type,
                        "role": role,
                        "template": tname,
                        "side": "left",
                        "prompt": left,
                        "seed": s,
                        "negatives": NEGATIVE_PROMPT,
                        "style": STYLE
                    })
                    all_prompts.append({
                        "pair_type": pair_type,
                        "role": role,
                        "template": tname,
                        "side": "right",
                        "prompt": right,
                        "seed": s,
                        "negatives": NEGATIVE_PROMPT,
                        "style": STYLE
                    })
    with open(out_json, "w") as f:
        json.dump(all_prompts, f, indent=2, ensure_ascii=False)
    print("Wrote", out_json)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", default="prompts_for_generation.json")
    args = parser.parse_args()
    make_prompts(args.pairs, args.out)

if __name__ == "__main__":
    main()