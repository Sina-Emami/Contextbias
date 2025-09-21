# eval_aggregate_report_hardcoded.py
# Hardcoded version: set GOLD_JSON, PRED_JSON, OUT_JSON below.

from __future__ import annotations
import json, re, sys
from collections import defaultdict
from typing import Dict, Any, Optional, List

# -------------------------
# 🔧 Hardcoded paths — edit these
# -------------------------
GOLD_JSON = "test_set/expected/expected_output.json"      # path to your gold-standard JSON
PRED_JSON = "test_set/descriptions/summary/summary_report.json"      # path to your analyzer's predicted JSON
OUT_JSON  = "test_set/metrics.json"   # where to write metrics (set to "" to skip writing)

# -------------------------
# Helpers
# -------------------------

def norm_key(s: str) -> str:
    """Robust, conservative normalization for labels/tokens/paths."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def is_nonempty_counts(d: Dict[str, Any]) -> bool:
    """True if d looks like a non-empty value_counts map with at least one positive integer."""
    if not isinstance(d, dict) or not d:
        return False
    for v in d.values():
        try:
            if int(v) > 0:
                return True
        except Exception:
            continue
    return False

def safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 0

# -------------------------
# Traversal & extraction
# -------------------------

def collect_value_counts(report: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """
    Walk the structured report and gather all 'value_counts' maps keyed by a human-readable dot path.
    Also converts each cohort's 'groups' array into a pseudo value_counts map {group_key: total_count}.
    Paths pointing to 'normalized' blocks are ignored.
    """
    out: Dict[str, Dict[str, int]] = {}

    def add_counts(path: str, raw_map: Dict[str, Any]):
        vc: Dict[str, int] = {}
        for k, v in raw_map.items():
            nk = norm_key(k)
            if nk == "":
                continue
            vc[nk] = vc.get(nk, 0) + safe_int(v)
        out[path] = vc

    def walk(obj: Any, path: str = "", cohort_label: Optional[str] = None):
        if isinstance(obj, dict):
            if "value_counts" in obj and isinstance(obj["value_counts"], dict):
                p = path
                if cohort_label:
                    p = f"cohort:{norm_key(cohort_label)}.{p}" if p else f"cohort:{norm_key(cohort_label)}"
                add_counts(f"{p}.value_counts" if p else "value_counts", obj["value_counts"])

            if "cohorts" in obj and isinstance(obj["cohorts"], list):
                for c in obj["cohorts"]:
                    if not isinstance(c, dict):
                        continue
                    clabel = c.get("cohort")
                    clabel_n = norm_key(clabel) if clabel else None
                    if "groups" in c and isinstance(c["groups"], list):
                        gmap: Dict[str, int] = {}
                        for g in c["groups"]:
                            if not isinstance(g, dict):
                                continue
                            gk = norm_key(g.get("group_key", ""))
                            cnt = safe_int(g.get("total_count", 0))
                            if gk:
                                gmap[gk] = gmap.get(gk, 0) + cnt
                        if gmap:
                            add_counts(f"cohort:{clabel_n}.groups", gmap)
                    for k, v in c.items():
                        if k in ("groups", "cohort"):
                            continue
                        walk(v, path=k if not path else f"{path}.{k}", cohort_label=clabel)

            for k, v in obj.items():
                if k in ("normalized", "cohorts"):
                    continue
                if isinstance(v, (dict, list)):
                    walk(v, path=k if not path else f"{path}.{k}", cohort_label=cohort_label)

        elif isinstance(obj, list):
            for i, it in enumerate(obj):
                walk(it, path=f"{path}[{i}]" if path else f"[{i}]", cohort_label=cohort_label)

    walk(report)
    return out

# -------------------------
# Metrics
# -------------------------

def set_metrics(gt: Dict[str, int], pr: Dict[str, int]) -> Dict[str, float]:
    gt_keys = set(k for k, v in gt.items() if safe_int(v) > 0)
    pr_keys = set(k for k, v in pr.items() if safe_int(v) > 0)
    tp = len(gt_keys & pr_keys)
    fp = len(pr_keys - gt_keys)
    fn = len(gt_keys - pr_keys)
    denom_p = tp + fp
    denom_r = tp + fn
    denom_acc = tp + fp + fn
    precision = tp / denom_p if denom_p else 0.0
    recall    = tp / denom_r if denom_r else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
    accuracy  = tp / denom_acc if denom_acc else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "tp": tp, "fp": fp, "fn": fn}

def count_metrics(gt: Dict[str, int], pr: Dict[str, int]) -> Dict[str, float]:
    keys = set(gt) | set(pr)
    tp = sum(min(safe_int(gt.get(k, 0)), safe_int(pr.get(k, 0))) for k in keys)
    fp = sum(max(0, safe_int(pr.get(k, 0)) - safe_int(gt.get(k, 0))) for k in keys)
    fn = sum(max(0, safe_int(gt.get(k, 0)) - safe_int(pr.get(k, 0))) for k in keys)
    denom_p = tp + fp
    denom_r = tp + fn
    denom_acc = tp + fp + fn
    precision = tp / denom_p if denom_p else 0.0
    recall    = tp / denom_r if denom_r else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
    accuracy  = tp / denom_acc if denom_acc else 0.0
    mae = sum(abs(safe_int(gt.get(k, 0)) - safe_int(pr.get(k, 0))) for k in keys) / (len(keys) if keys else 1)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "fn": fn, "mae": mae}

def aggregate_macro(metrics_list: List[Dict[str, float]], fields=("precision","recall","f1","accuracy")) -> Dict[str, float]:
    if not metrics_list:
        return {k: 0.0 for k in fields}
    return {k: sum(m.get(k, 0.0) for m in metrics_list) / len(metrics_list) for k in fields}

def aggregate_micro(conf_list: List[Dict[str, float]]) -> Dict[str, float]:
    TP = sum(int(m.get("tp", 0)) for m in conf_list)
    FP = sum(int(m.get("fp", 0)) for m in conf_list)
    FN = sum(int(m.get("fn", 0)) for m in conf_list)
    denom_p = TP + FP
    denom_r = TP + FN
    denom_acc = TP + FP + FN
    precision = TP / denom_p if denom_p else 0.0
    recall    = TP / denom_r if denom_r else 0.0
    f1        = (2 * precision * recall) / (precision + recall) if precision + recall else 0.0
    accuracy  = TP / denom_acc if denom_acc else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, "tp": TP, "fp": FP, "fn": FN}

# -------------------------
# Evaluation driver
# -------------------------

def evaluate(gold: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    gt_maps = collect_value_counts(gold)
    pr_maps = collect_value_counts(pred)

    per_path = []
    set_confs, cnt_confs = [], []

    for path, gt_counts in sorted(gt_maps.items()):
        if not is_nonempty_counts(gt_counts):
            continue
        pr_counts = pr_maps.get(path, {})

        sm = set_metrics(gt_counts, pr_counts)
        cm = count_metrics(gt_counts, pr_counts)
        per_path.append({
            "path": path,
            "set_metrics": sm,
            "count_metrics": cm,
            "gt_total": int(sum(safe_int(v) for v in gt_counts.values())),
            "pr_total": int(sum(safe_int(v) for v in pr_counts.values())),
            "gt_keys": sorted([k for k, v in gt_counts.items() if safe_int(v) > 0]),
            "pr_keys": sorted([k for k, v in pr_counts.items() if safe_int(v) > 0])
        })
        set_confs.append(sm)
        cnt_confs.append(cm)

    results = {
        "summary": {
            "num_paths_scored": len(per_path),
            "set_micro": aggregate_micro(set_confs),
            "set_macro": aggregate_macro(set_confs),
            "count_micro": aggregate_micro(cnt_confs),
            "count_macro": aggregate_macro(cnt_confs),
        },
        "details": per_path
    }
    return results

# -------------------------
# Main
# -------------------------

def main():
    try:
        with open(GOLD_JSON, "r", encoding="utf-8") as f:
            gold = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load GOLD_JSON='{GOLD_JSON}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(PRED_JSON, "r", encoding="utf-8") as f:
            pred = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load PRED_JSON='{PRED_JSON}': {e}", file=sys.stderr)
        sys.exit(1)

    metrics = evaluate(gold, pred)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if OUT_JSON:
        try:
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] Metrics written to: {OUT_JSON}")
        except Exception as e:
            print(f"[WARN] Could not write metrics to {OUT_JSON}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
