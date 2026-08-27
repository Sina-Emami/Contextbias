import argparse
import json
from difflib import SequenceMatcher

from tqdm import tqdm

from llm_client import call_llm

RELEVANCE_MIN = 4
NEUTRALITY_MIN = 4
CONFOUND_MAX = 2
DEDUPE_SIMILARITY = 0.88
MAX_ITEMS_PER_BUCKET = 12

SCORE_PROMPT = """
Role: "{role}"
Bucket: "{bucket}"
Item: "{item}"
Rationale: "{rationale}"

Score this item on three integer scales (1-5), returned as JSON:
{{"relevance": int, "neutrality": int, "confound": int}}

- relevance: how strongly the item fits the bucket (typical for "related", orthogonal for
  "unrelated").
- neutrality: 5 means no demographic or otherwise sensitive leakage.
- confound: 1 means minimal risk of implying gender/status/etc. through the item; 5 means high risk.

Return JSON only.
""".strip()


def _item_text(item: dict) -> str:
    if "item" in item:
        return item["item"]
    if "action" in item and "location" in item:
        return f"{item['action']} in {item['location']}"
    return str(item)


def score_item(role: str, bucket: str, item: dict, mock_mode: bool = False) -> dict:
    if mock_mode:
        return {"relevance": 4, "neutrality": 4, "confound": 2}

    prompt = SCORE_PROMPT.format(
        role=role, bucket=bucket, item=_item_text(item), rationale=item.get("rationale", "")
    )
    try:
        response = call_llm(prompt, max_tokens=500)
        return json.loads(response)
    except Exception as exc:
        print(f"Failed to score '{_item_text(item)}' for {role}/{bucket}: {exc}")
        return {"relevance": 3, "neutrality": 3, "confound": 3}


def passes_thresholds(score: dict) -> bool:
    return (
        score["relevance"] >= RELEVANCE_MIN
        and score["neutrality"] >= NEUTRALITY_MIN
        and score["confound"] <= CONFOUND_MAX
    )


def dedupe(items: list[dict], threshold: float = DEDUPE_SIMILARITY) -> list[dict]:
    kept: list[dict] = []
    kept_texts: list[str] = []
    for item in items:
        text = _item_text(item)
        if any(SequenceMatcher(None, text, seen).ratio() >= threshold for seen in kept_texts):
            continue
        kept.append(item)
        kept_texts.append(text)
    return kept


def judge_bucket(role: str, bucket: str, items: list[dict], apply_filter: bool, mock_mode: bool) -> list[dict]:
    scored = []
    for item in items:
        score = score_item(role, bucket, item, mock_mode=mock_mode)
        accept = passes_thresholds(score) if apply_filter else True
        scored.append({**item, "score": score, "accept": accept})

    accepted = dedupe([item for item in scored if item["accept"]])
    return accepted[:MAX_ITEMS_PER_BUCKET]


def main(args: argparse.Namespace) -> None:
    with open(args.infile) as f:
        candidates = json.load(f)

    filtered = {}
    for role, buckets in tqdm(candidates.items(), desc="roles"):
        filtered[role] = {
            bucket_name: judge_bucket(role, bucket_name, buckets.get(bucket_name, []), args.filter, args.mock)
            for bucket_name in ("related", "unrelated")
        }

    with open(args.out, "w") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score and filter candidate action-location pairs.")
    parser.add_argument("--infile", required=True, help="Path to candidates.json")
    parser.add_argument("--out", default="context_bank.json", help="Output file")
    parser.add_argument("--model", default=None, help="Unused; kept for CLI compatibility")
    parser.add_argument("--filter", action="store_true", help="Apply relevance/neutrality/confound thresholds")
    parser.add_argument("--mock", action="store_true", help="Use fallback scores instead of calling the LLM")
    main(parser.parse_args())
