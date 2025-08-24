import json
from pathlib import Path
from typing import Dict, Any, Iterable
from crewai.tools import tool

# ---------- helpers (deterministic extraction/aggregation) ----------
def _canon(x: str | None) -> str:
    if not x:
        return ""
    return " ".join(x.lower().replace("_", " ").replace("-", " ").split())

def _inc(d: Dict[str, int], k: str, by: int = 1):
    if not k:
        return
    d[k] = d.get(k, 0) + by

def _inc_nested(root: Dict[str, Any], typ: str, attr: str, val: str, by: int = 1):
    if not (typ and attr and val):
        return
    tt = root.setdefault(typ, {})
    aa = tt.setdefault(attr, {})
    aa[val] = aa.get(val, 0) + by

def _extract_per_image(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically summarize one ImageAuditRecord dict into countable buckets."""
    out: Dict[str, Any] = {
        "meta": {"image_id": rec.get("image_id")},
        "objects_by_name": {},
        "environment_by_label": {},
        "object_attribute_counts": {},
        "environment_attribute_counts": {},
        "people_meta": {"people_in_image": len(rec.get("people", []) or [])},
        "people_attribute_counts": {},
        "people_detail_phrases": {},
        "detail_phrases": {},
    }

    # objects
    for obj in rec.get("objects", []) or []:
        typ = _canon(obj.get("name_canonical") or obj.get("name_raw"))
        _inc(out["objects_by_name"], typ, 1)
        for c in set(_canon(c) for c in (obj.get("colors") or []) if c):
            _inc_nested(out["object_attribute_counts"], typ, "color", c, 1)
            _inc(out["detail_phrases"], f"{typ}:{c}", 1)
        for m in set(_canon(m) for m in (obj.get("materials") or []) if m):
            _inc_nested(out["object_attribute_counts"], typ, "material", m, 1)
            _inc(out["detail_phrases"], f"{typ}:{m}", 1)
        for k, v in (obj.get("attributes") or {}).items():
            kk, vv = _canon(k), _canon(str(v))
            _inc_nested(out["object_attribute_counts"], typ, f"attributes:{kk}", vv, 1)

    # environment
    for e in ((rec.get("setting") or {}).get("environment_elements") or []):
        lab = _canon(e.get("label_canonical") or e.get("label_raw"))
        _inc(out["environment_by_label"], lab, 1)
        for c in set(_canon(c) for c in (e.get("colors") or []) if c):
            _inc_nested(out["environment_attribute_counts"], lab, "color", c, 1)
            _inc(out["detail_phrases"], f"{lab}:{c}", 1)
        for m in set(_canon(m) for m in (e.get("materials") or []) if m):
            _inc_nested(out["environment_attribute_counts"], lab, "material", m, 1)
            _inc(out["detail_phrases"], f"{lab}:{m}", 1)
        for k, v in (e.get("attributes") or {}).items():
            kk, vv = _canon(k), _canon(str(v))
            _inc_nested(out["environment_attribute_counts"], lab, f"attributes:{kk}", vv, 1)

    # people
    ppl = rec.get("people", []) or []
    pac = out["people_attribute_counts"]

    def bump(group: str, val: str | None):
        if not val:
            return
        g = pac.setdefault(group, {})
        key = _canon(val)
        _inc(g, key, 1)

    for p in ppl:
        for key in [
            "gender_presentation", "age_bucket", "orientation", "posture",
            "position_depth", "position_horizontal", "skin_tone_label",
            "race_ethnicity_label", "visible_tattoos", "facial_hair", "eyewear",
        ]:
            bump(key, p.get(key))
        for a in set(_canon(a) for a in (p.get("accessories") or []) if a):
            bump("accessories", a)
        for c in set(_canon(c) for c in (p.get("clothing_items") or []) if c):
            bump("clothing_items", c)

        def pdp(tag: str, val: str | None):
            v = _canon(val)
            if v:
                _inc(out["people_detail_phrases"], f"{tag}:{v}", 1)

        pdp("gender", p.get("gender_presentation"))
        pdp("age", p.get("age_bucket"))
        pdp("orientation", p.get("orientation"))
        pdp("posture", p.get("posture"))
        pdp("position_depth", p.get("position_depth"))
        pdp("position_horizontal", p.get("position_horizontal"))
        pdp("skin_tone", p.get("skin_tone_label"))
        pdp("race_ethnicity", p.get("race_ethnicity_label"))
        pdp("visible_tattoos", p.get("visible_tattoos"))
        pdp("facial_hair", p.get("facial_hair"))
        pdp("eyewear", p.get("eyewear"))
        for a in set(_canon(a) for a in (p.get("accessories") or []) if a):
            pdp("accessory", a)
        for c in set(_canon(c) for c in (p.get("clothing_items") or []) if c):
            pdp("clothing", c)

    return out

def _merge_nested(dst: Dict[str, Any], src: Dict[str, Any]):
    for typ, attrs in (src or {}).items():
        tt = dst.setdefault(typ, {})
        for attr, vals in attrs.items():
            tv = tt.setdefault(attr, {})
            for val, cnt in vals.items():
                tv[val] = tv.get(val, 0) + int(cnt)

def _split(counter: Dict[str, int]) -> tuple[Dict[str, int], Dict[str, int]]:
    reps = {k: v for k, v in counter.items() if v >= 2}
    singles = {k: v for k, v in counter.items() if v == 1}
    return reps, singles

# ---------- tools used by the memory-enabled agent ----------
@tool("ingest_description")
def ingest_description(description_path: str, state_path: str) -> str:
    """
    Parse a single ImageAuditRecord JSON file and upsert its deterministic
    per-image summary into an aggregation state JSON (idempotent upsert).
    """
    desc = json.loads(Path(description_path).read_text(encoding="utf-8"))
    per_image = _extract_per_image(desc)
    iid = per_image["meta"]["image_id"] or Path(description_path).stem

    state_file = Path(state_path)
    if state_file.exists():
        S = json.loads(state_file.read_text(encoding="utf-8"))
    else:
        S = {"per_image": {}, "final": None}

    S["per_image"][iid] = per_image     # overwrite if exists (idempotent)
    S["final"] = None                   # invalidate cached final
    state_file.write_text(json.dumps(S, indent=2, ensure_ascii=False), encoding="utf-8")
    return "ok"

@tool("finalize_summary")
def finalize_summary(state_path: str, out_path: str | None = None) -> Dict[str, Any]:
    """
    Recompute the final repetition summary across all upserted images
    and optionally write it to 'out_path'.
    """
    S = json.loads(Path(state_path).read_text(encoding="utf-8"))
    imgs = S.get("per_image", {})
    totals_objects: Dict[str, int] = {}
    totals_env: Dict[str, int] = {}
    people_total = 0
    people_per_image: Dict[str, int] = {}
    rep_detail: Dict[str, int] = {}
    rep_people_detail: Dict[str, int] = {}
    obj_attr: Dict[str, Any] = {}
    env_attr: Dict[str, Any] = {}
    ppl_attrs: Dict[str, Dict[str, int]] = {}

    for iid, im in imgs.items():
        for k, v in im["objects_by_name"].items(): _inc(totals_objects, k, int(v))
        for k, v in im["environment_by_label"].items(): _inc(totals_env, k, int(v))
        people_total += int(im["people_meta"]["people_in_image"])
        people_per_image[iid] = int(im["people_meta"]["people_in_image"])
        for k, v in im["detail_phrases"].items(): _inc(rep_detail, k, int(v))
        for k, v in im["people_detail_phrases"].items(): _inc(rep_people_detail, k, int(v))
        _merge_nested(obj_attr, im["object_attribute_counts"])
        _merge_nested(env_attr, im["environment_attribute_counts"])
        for grp, mp in (im["people_attribute_counts"] or {}).items():
            tgt = ppl_attrs.setdefault(grp, {})
            for val, cnt in mp.items(): _inc(tgt, val, int(cnt))

    rep_objs, sing_objs = _split(totals_objects)
    rep_env, sing_env = _split(totals_env)
    sing_detail = {k: v for k, v in rep_detail.items() if v == 1}
    sing_people_detail = {k: v for k, v in rep_people_detail.items() if v == 1}

    final = {
        "meta": {"num_images": len(imgs), "image_ids": sorted(imgs.keys())},
        "totals": {
            "objects": totals_objects,
            "environment_elements": totals_env,
            "people": {"total_people_detected": people_total, "people_per_image": people_per_image},
        },
        "repeats": {
            "objects_by_name": rep_objs,
            "environment_by_label": rep_env,
            "object_attribute_counts": obj_attr,
            "environment_attribute_counts": env_attr,
            "detail_phrases": rep_detail,
            "people_attribute_counts": ppl_attrs,
            "people_detail_phrases": rep_people_detail,
        },
        "singles": {
            "objects_by_name": sing_objs,
            "environment_by_label": sing_env,
            "object_attribute_counts": {},
            "environment_attribute_counts": {},
            "detail_phrases": sing_detail,
            "people_attribute_counts": {},
            "people_detail_phrases": sing_people_detail,
        },
    }

    if out_path:
        Path(out_path).write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    S["final"] = final
    Path(state_path).write_text(json.dumps(S, indent=2, ensure_ascii=False), encoding="utf-8")
    return final
