import json
import argparse
from tqdm import tqdm
from llm_client import call_llm
from rapidfuzz import fuzz

# thresholds
RELEVANCE_MIN = 4
NEUTRALITY_MIN = 4
CONFOUND_MAX = 2

JUDGE_PROMPT_TEMPLATE = """
For the ROLE "{role}", axis "{axis}" and item "{item}" with rationale "{rationale}":
Score the following three numbers (integers 1-5) in JSON:
{{"relevance": int, "neutrality": int, "confound": int}}
Definitions:
- relevance: how strongly this item is typical for the role (for related lists).
- orthogonality: for unrelated lists we interpret 'relevance' as 'orthogonality' (higher = more orthogonal).
- neutrality: 5 = no demographic or sensitive leakage.
- confound: 1 = minimal risk; 5 = high confound (e.g., implies gender/status via prop).
Return JSON only.
""".strip()

def score_item(role, axis, item, rationale, model=None, mock_mode=False):
    if mock_mode:
        return {
            "relevance": 4,
            "neutrality": 4,
            "confound": 2
        }
    
    prompt = JUDGE_PROMPT_TEMPLATE.format(role=role, axis=axis, item=item, rationale=rationale)
    try:
        txt = call_llm(prompt, max_tokens=500)
        try:
            out = json.loads(txt)
        except:
            start = txt.find("{")
            end = txt.rfind("}")+1
            out = json.loads(txt[start:end])
        return out
    except Exception as e:
        print(f"Error calling LLM for {role}/{axis}/{item}: {e}")
        return {
            "relevance": 3,
            "neutrality": 3,
            "confound": 3
        }

def dedupe_list(items, threshold=90):
    kept = []
    for itm in items:
        if "item" in itm:
            text = itm["item"]
        elif "action" in itm and "location" in itm:
            text = f"{itm['action']} in {itm['location']}"
        else:
            text = str(itm)
        if any(
            ("item" in k and fuzz.ratio(text, k["item"]) >= threshold) or
            ("action" in k and "location" in k and fuzz.ratio(text, f"{k['action']} in {k['location']}") >= threshold)
            for k in kept
        ):
            continue
        kept.append(itm)
    return kept

def main(args):
    with open(args.infile, "r") as f:
        candidates = json.load(f)
    filtered = {}
    for role, axes in tqdm(candidates.items(), desc="roles"):
        filtered[role] = {"related": [], "unrelated": []}
        for bucket_name in ["related", "unrelated"]:
            items = axes.get(bucket_name, [])
            scored = []
            for it in items:
                if "item" in it:
                    item_text = it["item"]
                elif "action" in it and "location" in it:
                    item_text = f"{it['action']} in {it['location']}"
                else:
                    item_text = str(it)
                rationale = it.get("rationale","")
                score = score_item(role, bucket_name, item_text, rationale, model=args.model, mock_mode=args.mock)
                it_out = {**it, "score": score}
                if args.filter:
                    rel_ok = score["relevance"] >= RELEVANCE_MIN if bucket_name=="related" else score["relevance"] >= RELEVANCE_MIN
                    neu_ok = score["neutrality"] >= NEUTRALITY_MIN
                    conf_ok = score["confound"] <= CONFOUND_MAX
                    it_out["accept"] = bool(rel_ok and neu_ok and conf_ok)
                else:
                    it_out["accept"] = True
                scored.append(it_out)
            accepted = [s for s in scored if s["accept"]]
            accepted = dedupe_list(accepted, threshold=88)
            filtered[role][bucket_name] = accepted[:12]
    with open(args.out, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--out", default="context_bank.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--filter", action="store_true", help="Enable filtering by thresholds (default: on)")
    parser.add_argument("--mock", action="store_true", help="Use mock scores instead of calling LLM API")
    args = parser.parse_args()
    main(args)