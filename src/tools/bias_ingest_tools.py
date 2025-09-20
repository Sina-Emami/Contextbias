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
    scene = rec.get("scene") or {}
    out: Dict[str, Any] = {
        "meta": {"image_id": rec.get("image_id")},
        "objects_by_name": {},
        "environment_by_label": {},
        "object_attribute_counts": {},
        "environment_attribute_counts": {},
        "people_meta": {"people_in_image": len(scene.get("people") or [])},
        "people_attribute_counts": {},
        "people_detail_phrases": {},
        "detail_phrases": {},
    }

    objects = scene.get("objects") or []
    for obj in objects:
        typ_raw = obj.get("type") or obj.get("subtype") or "unknown"
        typ = _canon(typ_raw)
        _inc(out["objects_by_name"], typ, 1)
        attrs = obj.get("attributes") or {}
        for c in set(_canon(c) for c in (attrs.get("colors") or []) if c):
            _inc_nested(out["object_attribute_counts"], typ, "color", c, 1)
            _inc(out["detail_phrases"], f"{typ}:color:{c}", 1)
        for m in set(_canon(m) for m in (attrs.get("material") or []) if m):
            _inc_nested(out["object_attribute_counts"], typ, "material", m, 1)
            _inc(out["detail_phrases"], f"{typ}:material:{m}", 1)
        for tex in set(_canon(t) for t in (attrs.get("texture") or []) if t):
            _inc_nested(out["object_attribute_counts"], typ, "texture", tex, 1)
        for fin in set(_canon(t) for t in (attrs.get("finish") or []) if t):
            _inc_nested(out["object_attribute_counts"], typ, "finish", fin, 1)
        size_class = _canon(attrs.get("size_class"))
        if size_class:
            _inc_nested(out["object_attribute_counts"], typ, "size_class", size_class, 1)
        for state in set(_canon(s) for s in (attrs.get("state") or []) if s):
            _inc_nested(out["object_attribute_counts"], typ, "state", state, 1)

    environment = scene.get("environment") or {}
    env_label = _canon(environment.get("location_type"))
    if env_label:
        _inc(out["environment_by_label"], env_label, 1)

    surfaces = environment.get("surfaces") or {}
    for group_name in ("walls", "floor", "ceiling"):
        details = surfaces.get(group_name) or []
        label = _canon(group_name)
        for detail in details:
            _inc(out["environment_by_label"], label, 1)
            for mat in set(_canon(m) for m in (detail.get("material") or []) if m):
                _inc_nested(out["environment_attribute_counts"], label, "material", mat, 1)
            for tex in set(_canon(t) for t in (detail.get("texture") or []) if t):
                _inc_nested(out["environment_attribute_counts"], label, "texture", tex, 1)
            for col in set(_canon(c) for c in (detail.get("color") or []) if c):
                _inc_nested(out["environment_attribute_counts"], label, "color", col, 1)
            for cond in set(_canon(c) for c in (detail.get("condition") or []) if c):
                _inc_nested(out["environment_attribute_counts"], label, "condition", cond, 1)

    ppl = scene.get("people") or []
    pac = out["people_attribute_counts"]
    detail = out["people_detail_phrases"]

    def bump(group: str, val: str | None):
        if not val:
            return
        key = _canon(val)
        if not key:
            return
        bucket = pac.setdefault(group, {})
        _inc(bucket, key, 1)

    def bump_detail(tag: str, val: str | None):
        key = _canon(val)
        if key:
            _inc(detail, f"{tag}:{key}", 1)

    for person in ppl:
        bump("gender_presentation", person.get("gender_presentation"))
        bump("age_range", person.get("age_range"))
        bump("orientation", person.get("orientation"))
        bump("gaze_direction", person.get("gaze_direction"))
        bump("occlusions", person.get("occlusions"))
        bump("skin_tone", person.get("skin_tone"))
        bump_detail("gender", person.get("gender_presentation"))
        bump_detail("age", person.get("age_range"))
        bump_detail("orientation", person.get("orientation"))
        bump_detail("gaze", person.get("gaze_direction"))
        bump_detail("skin_tone", person.get("skin_tone"))

        hair = person.get("hair") or {}
        bump("hair_present", hair.get("present"))
        for style in set(_canon(s) for s in (hair.get("style") or []) if s):
            bump("hair_style", style)
        for color in set(_canon(c) for c in (hair.get("color") or []) if c):
            bump("hair_color", color)

        facial = person.get("facial_hair") or {}
        bump("facial_hair_present", facial.get("present"))
        for style in set(_canon(s) for s in (facial.get("style") or []) if s):
            bump("facial_hair_style", style)

        eyewear = person.get("eyewear") or {}
        bump("eyewear_present", eyewear.get("present"))
        for typ in set(_canon(t) for t in (eyewear.get("type") or []) if t):
            bump("eyewear_type", typ)

        head = person.get("head_covering") or {}
        bump("head_covering_present", head.get("present"))
        for typ in set(_canon(t) for t in (head.get("type") or []) if t):
            bump("head_covering_type", typ)

        for pose in set(_canon(pose) for pose in (person.get("pose") or []) if pose):
            bump("pose", pose)
        for act in set(_canon(act) for act in (person.get("activities") or []) if act):
            bump("activities", act)

        clothing_items = person.get("clothing") or []
        for garment in clothing_items:
            typ = _canon(garment.get("garment_type"))
            if typ:
                bump("clothing_type", typ)
            for color in set(_canon(c) for c in (garment.get("color") or []) if c):
                bump("clothing_color", color)
            for material in set(_canon(m) for m in (garment.get("material") or []) if m):
                bump("clothing_material", material)
            for pattern in set(_canon(p) for p in (garment.get("pattern") or []) if p):
                bump("clothing_pattern", pattern)

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
