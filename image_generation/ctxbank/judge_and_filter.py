# ctxbank/judge_and_filter.py
import json
import argparse
from tqdm import tqdm
from image_generation.ctxbank.llm_client import call_llm
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

def score_item(role, axis, item, rationale, model=None):
    prompt = JUDGE_PROMPT_TEMPLATE.format(role=role, axis=axis, item=item, rationale=rationale)
    txt = call_llm(prompt, max_tokens=30000)
    try:
        out = json.loads(txt)
    except:
        # try extract
        start = txt.find("{")
        end = txt.rfind("}")+1
        out = json.loads(txt[start:end])
    return out

def dedupe_list(items, threshold=90):
    kept = []
    for itm in items:
        text = itm["item"]
        if any(fuzz.ratio(text, k["item"]) >= threshold for k in kept):
            continue
        kept.append(itm)
    return kept

def main(args):
    with open(args.infile, "r") as f:
        candidates = json.load(f)
    filtered = {}
    for role, axes in tqdm(candidates.items(), desc="roles"):
        filtered[role] = {}
        for axis, buckets in axes.items():
            filtered[role].setdefault(axis, {"related": [], "unrelated": []})
            for bucket_name in ["related", "unrelated"]:
                items = buckets.get(bucket_name, [])
                scored = []
                for it in items:
                    item_text = it.get("item")
                    rationale = it.get("rationale","")
                    score = score_item(role, axis, item_text, rationale, model=args.model)
                    it_out = {**it, "score": score}
                    # apply thresholds depending on bucket
                    rel_ok = score["relevance"] >= RELEVANCE_MIN if bucket_name=="related" else score["relevance"] >= RELEVANCE_MIN
                    neu_ok = score["neutrality"] >= NEUTRALITY_MIN
                    conf_ok = score["confound"] <= CONFOUND_MAX
                    it_out["accept"] = bool(rel_ok and neu_ok and conf_ok)
                    scored.append(it_out)
                # keep accepted items, dedupe, cap to 12
                accepted = [s for s in scored if s["accept"]]
                accepted = dedupe_list(accepted, threshold=88)
                filtered[role][axis][bucket_name] = accepted[:12]
    with open(args.out, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print("Wrote", args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--out", default="context_bank.json")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    main(args)