"""
Unified role aggregation utilities.

This module collects the logic for normalising cohort/dimension/label tokens and
provides two command-line entry points:

1. Aggregate per-role counts across all dataset contexts and write the merged CSVs
   to ``dataset/role_count_aggregation/<role>.csv``.
2. (Optional) Build a single ``dataset/general_attributes_rollup.csv`` summarising
   high-level attributes across every role.

Usage
=====

Aggregate per-role counts::

    python -m decription_pipeline.data_processing.role_rollup

Aggregate per-role counts and build the general attribute rollup::

    python -m decription_pipeline.data_processing.role_rollup --general
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

DATASET_ROOT = Path("dataset")
ROLE_OUTPUT_DIR = DATASET_ROOT / "role_count_aggregation"
GENERAL_OUTPUT_PATH = DATASET_ROOT / "general_attributes_rollup.csv"

# ---------------------------------------------------------------------------
# Normalisation tables
# ---------------------------------------------------------------------------

KEY_NORMALISATION: Dict[str, str] = {
    "demographics.gender_presentation": "gender_presentation",
    "demographics_gender_presentation": "gender_presentation",
    "demographics.age_range": "age_range",
    "demographics_age_range": "age_range",
    "demographics.role_hint": "role_hint",
    "demographics_role_hint": "role_hint",
    "demographics.skin_tone": "skin_tone",
    "demographics_skin_tone": "skin_tone",
    "appearance.eyewear.frame_color": "eyewear_frame_color",
    "appearance.eyewear_present": "eyewear_present",
    "appearance_eyewear_frame_color": "eyewear_frame_color",
    "appearance.eyewear.present": "eyewear_present",
    "appearance_eyewear_present": "eyewear_present",
    "appearance.eyewear.type": "eyewear_type",
    "appearance_eyewear_type": "eyewear_type",
    "appearance.facial_hair.color": "facial_hair_color",
    "appearance_facial_hair_color": "facial_hair_color",
    "appearance.facial_hair.present": "facial_hair_present",
    "appearance_facial_hair_present": "facial_hair_present",
    "appearance_facial_hair_presence": "facial_hair_present",
    "appearance.facial_hair.style": "facial_hair_style",
    "appearance_facial_hair_style": "facial_hair_style",
    "appearance.hair.color": "hair_color",
    "appearance.hair_color": "hair_color",
    "appearance_hair_color": "hair_color",
    "appearance.hair.present": "hair_present",
    "appearance.hair_present": "hair_present",
    "appearance_hair_present": "hair_present",
    "appearance.hair.style": "hair_style",
    "appearance.hair_style": "hair_style",
    "appearance_hair_style": "hair_style",
    "appearance.head_covering.present": "head_covering_present",
    "appearance.head_covering_present": "head_covering_present",
    "appearance_head_covering_present": "head_covering_present",
    "appearance.head_covering.color": "head_covering_color",
    "appearance.head_covering_color": "head_covering_color",
    "appearance_head_covering_color": "head_covering_color",
    "appearance.head_covering.type": "head_covering_type",
    "appearance.head_covering_type": "head_covering_type",
    "appearance_head_covering_type": "head_covering_type",
    "clothing.by_garment_type": "clothing_garment",
    "clothing_by_garment_type": "clothing_garment",
    "clothing_garment_type": "clothing_garment",
    "clothing_type.item": "clothing_item",
    "clothing_type_item": "clothing_item",
    "clothingtypeitem": "clothing_item",
    "emotion.feeling": "emotional_state",
    "emotion_feeling": "emotional_state",
    "emotionfeeling": "emotional_state",
    "pose_activity.pose": "pose_activity_pose",
    "pose_activity.gaze_direction": "pose_activity_gaze_direction",
    "pose_activity.occlusions": "pose_activity_occlusions",
    "pose_activity.activities": "pose_activity_activities",
}

PREFIX_NORMALISATION = [
    ("clothing.by_garment_type.", "clothing_garment"),
    ("clothing_by_garment_type_", "clothing_garment"),
    ("clothing_by_garment_type.", "clothing_garment"),
    ("clothing_garment_type_", "clothing_garment"),
]

VALUE_NORMALISATION: Dict[str, str] = {
    "background.center": "background center",
    "background.left": "background left",
    "background.right": "background right",
    "foreground.center": "foreground center",
    "foreground.left": "foreground left",
    "foreground.right": "foreground right",
    "midground.center": "midground center",
    "midground.left": "midground left",
    "midground.right": "midground right",
    "head_wear": "headwear",
    "lamp_post": "lamppost",
    "neck_tie": "necktie",
    "tree_line": "treeline",
    "t_shirt": "t-shirt",
    "tshirt": "t-shirt",
    "tee_shirt": "t-shirt",
}

PALETTE_COLOURS = {
    "black",
    "white",
    "brown",
    "blue",
    "red",
    "green",
    "orange",
    "yellow",
    "purple",
    "pink",
    "cyan",
    "magenta",
    "grey",
    "gray",
    "beige",
    "silver",
    "gold",
    "ghostwhite",
}

PRESENCE_POSITIVE = {"present", "yes", "y", "true", "with", "wearing", "visible", "on"}
PRESENCE_NEGATIVE = {"absent", "no", "n", "false", "without", "none", "not", "missing"}
UNKNOWN_TOKENS = {"unknown", "unk", "n/a", "na"}

PLANE_TOKENS = {"foreground", "midground", "background"}
SIDE_TOKENS = {"left", "center", "right"}

OBJECT_TYPE_MAP = {
    "cleaning accessory": "cleaning tool",
    "cleaning accessories": "cleaning tool",
    "cleaning equipment": "cleaning tool",
    "cleaning tools": "cleaning tool",
    "cleaning tool": "cleaning tool",
    "backplash": "backsplash",
    "backplash typo token retained": "backsplash",
}
OBJECT_SUFFIXES = (" group", " cluster", " set", " objects", " object", " total", " items", " item")

CLOTHING_MAP = {
    "neck accessory": "necklace",
    "neck_accessory": "necklace",
    "neck chain": "necklace",
    "vest or suspenders": "vest",
    "t shirt": "t-shirt",
    "t_shirt": "t-shirt",
    "tshirt": "t-shirt",
    "tee shirt": "t-shirt",
}
SHIRT_PATTERNS = [
    r"^(.*)work\s+shirt$",
    r"^(.*)short\s*sleeve(d)?\s*work\s*shirt$",
    r"^(.*)short\s*sleeve(d)?\s*shirt$",
    r"^(.*)polo\s*shirt$",
]

# General attribute aggregation configuration
GENERAL_ATTRIBUTES = [
    "Mood",
    "LightingColorTemperature",
    "AtmosphereContrastLevel",
    "SaturationLevel",
    "DominantPalette",
    "AestheticQualities",
    "IndoorOutdoor",
    "Weather",
    "TimeOfDayHint",
    "Perspective",
    "Framing",
    "FocalLength",
    "DepthOfField",
    "CameraAngle",
    "Directionality",
    "Hardness",
    "Shadows",
    "SourcesCount",
    "SourcesType",
    "GenderPresentation",
    "SkinTone",
    "AgeRange",
    "GazeDirection",
    "OrientationType",
    "FaceEmotion",
    "Hairstyle",
    "HairColor",
    "OcclusionState",
    "eyewear_present",
    "head_covering_present",
    "facial_hair_present",
    "safety_gear_present",
    "visible_text_present",
    "vehicle_present",
    "animal_present",
    "tool_present",
    "multiple_people_present",
    "hair_present",
]

GENERAL_ATTRIBUTE_MAP: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("atmosphere", "mood"): ("Mood", "atmosphere"),
    ("atmosphere", "mood_distribution"): ("Mood", "atmosphere"),
    ("atmosphere", "color_temperature"): ("LightingColorTemperature", "atmosphere"),
    ("atmosphere", "lighting_color_temperature"): ("LightingColorTemperature", "atmosphere"),
    ("atmosphere", "lighting_profile_color_temperature"): ("LightingColorTemperature", "atmosphere"),
    ("lighting", "color_temperature"): ("LightingColorTemperature", "lighting"),
    ("lighting", "lighting_profile_color_temperature"): ("LightingColorTemperature", "lighting"),
    ("atmosphere", "contrast_level"): ("AtmosphereContrastLevel", "atmosphere"),
    ("atmosphere", "lighting_profile_contrast_level"): ("AtmosphereContrastLevel", "atmosphere"),
    ("lighting", "contrast_level"): ("AtmosphereContrastLevel", "lighting"),
    ("lighting", "lighting_profile_contrast_level"): ("AtmosphereContrastLevel", "lighting"),
    ("atmosphere", "saturation_level"): ("SaturationLevel", "atmosphere"),
    ("atmosphere", "lighting_profile_saturation_level"): ("SaturationLevel", "atmosphere"),
    ("lighting", "saturation_level"): ("SaturationLevel", "lighting"),
    ("lighting", "lighting_profile_saturation_level"): ("SaturationLevel", "lighting"),
    ("atmosphere", "dominant_palette"): ("DominantPalette", "atmosphere"),
    ("atmosphere", "dominant_palette_mapped"): ("DominantPalette", "atmosphere"),
    ("atmosphere", "palette"): ("DominantPalette", "atmosphere"),
    ("atmosphere", "palette_color"): ("DominantPalette", "atmosphere"),
    ("atmosphere", "palette_colors"): ("DominantPalette", "atmosphere"),
    ("atmosphere", "palette_detail"): ("DominantPalette", "atmosphere"),
    ("atmosphere", "aesthetic_qualities"): ("AestheticQualities", "atmosphere"),
    ("atmosphere", "aesthetic_quality"): ("AestheticQualities", "atmosphere"),
    ("atmosphere", "lighting_profile_aesthetic_qualities"): ("AestheticQualities", "atmosphere"),
    ("environment", "indoor_outdoor"): ("IndoorOutdoor", "environment"),
    ("environment", "weather"): ("Weather", "environment"),
    ("environment", "time_of_day_hint"): ("TimeOfDayHint", "environment"),
    ("camera", "perspective"): ("Perspective", "camera"),
    ("camera", "framing"): ("Framing", "camera"),
    ("camera", "focal_length"): ("FocalLength", "camera"),
    ("camera", "depth_of_field"): ("DepthOfField", "camera"),
    ("camera", "angle"): ("CameraAngle", "camera"),
    ("lighting", "directionality"): ("Directionality", "lighting"),
    ("lighting", "hardness"): ("Hardness", "lighting"),
    ("lighting", "shadows"): ("Shadows", "lighting"),
    ("lighting", "sources_count"): ("SourcesCount", "lighting"),
    ("lighting", "sources_type"): ("SourcesType", "lighting"),
    ("people", "gender_presentation"): ("GenderPresentation", "people"),
    ("people", "skin_tone"): ("SkinTone", "people"),
    ("people", "age_range"): ("AgeRange", "people"),
    ("people", "gaze_direction"): ("GazeDirection", "people"),
    ("people", "orientation"): ("OrientationType", "people"),
    ("people", "pose_activity_orientation"): ("OrientationType", "people"),
    ("people", "face_emotion"): ("FaceEmotion", "people"),
    ("people", "hair_style"): ("Hairstyle", "people"),
    ("people", "hair_color"): ("HairColor", "people"),
    ("people", "occlusions"): ("OcclusionState", "people"),
    ("people", "pose_activity_occlusions"): ("OcclusionState", "people"),
    ("people", "eyewear_present"): ("eyewear_present", "people"),
    ("people", "head_covering_present"): ("head_covering_present", "people"),
    ("people", "facial_hair_present"): ("facial_hair_present", "people"),
    ("people", "facial_hair_presence"): ("facial_hair_present", "people"),
    ("people", "hair_present"): ("hair_present", "people"),
    ("people", "safety_gear_present"): ("safety_gear_present", "people"),
    ("texts", "visible_text_present"): ("visible_text_present", "texts"),
    ("objects", "vehicle_present"): ("vehicle_present", "objects"),
    ("objects", "animal_present"): ("animal_present", "objects"),
    ("objects", "tool_present"): ("tool_present", "objects"),
    ("people", "multiple_people_present"): ("multiple_people_present", "people"),
    ("*", "gender_presentation"): ("GenderPresentation", "people"),
    ("*", "hair_present"): ("hair_present", "people"),
    ("*", "eyewear_present"): ("eyewear_present", "people"),
    ("*", "head_covering_present"): ("head_covering_present", "people"),
    ("*", "facial_hair_present"): ("facial_hair_present", "people"),
}

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    result = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text))
    parts = [part for part in result.split("-") if part]
    return "-".join(parts) or "value"


def slugify_name(text: str) -> str:
    """Normalized slug for filenames."""
    return slugify(text).replace("_", "-")


def norm_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_dimension_key(raw_key: str) -> str:
    if raw_key is None:
        return "unknown"
    key = raw_key.strip()
    if not key:
        return "unknown"
    key_lower = key.lower()
    if key_lower in KEY_NORMALISATION:
        return KEY_NORMALISATION[key_lower]
    if key in KEY_NORMALISATION:
        return KEY_NORMALISATION[key]
    for prefix, canonical in PREFIX_NORMALISATION:
        if key_lower.startswith(prefix) or key.startswith(prefix):
            return canonical
    normalized = re.sub(r"[^0-9a-z]+", "_", key_lower).strip("_")
    if normalized in KEY_NORMALISATION:
        return KEY_NORMALISATION[normalized]
    for prefix, canonical in PREFIX_NORMALISATION:
        if normalized.startswith(prefix):
            return canonical
    return normalized or "unknown"


def normalize_dimension_name(cohort: str, raw_key: str) -> str:
    cohort_token = normalize_cohort(cohort)
    key_token = normalize_dimension_key(raw_key)
    return f"{cohort_token}.{key_token}"


def normalize_cohort(raw: str) -> str:
    return slugify(raw or "unknown")


def normalize_presence_label(label: str) -> str:
    if any(tok in label for tok in UNKNOWN_TOKENS):
        return "unknown"
    toks = set(re.findall(r"[a-z0-9]+", label))
    if toks & PRESENCE_POSITIVE:
        return "yes"
    if toks & PRESENCE_NEGATIVE:
        return "no"
    if "without" in label or "absent" in label:
        return "no"
    if "with" in label or "present" in label:
        return "yes"
    return "unknown"


def normalize_gender_label(label: str) -> str:
    if "female" in label:
        return "female"
    if "male" in label:
        return "male"
    if "unknown" in label:
        return "unknown"
    return "unknown"


def normalize_color_temperature_label(label: str) -> str:
    if "cool" in label:
        return "cool"
    if "warm" in label:
        return "warm"
    if "neutral" in label:
        return "neutral"
    if "unknown" in label:
        return "unknown"
    return label


def normalize_contrast_label(label: str) -> str:
    if "high" in label:
        return "high"
    if "low" in label:
        return "low"
    if "medium" in label or "moderate" in label:
        return "medium"
    if "unknown" in label:
        return "unknown"
    return label


def normalize_palette_label(tokens: Iterable[str]) -> str:
    for token in tokens:
        if token in PALETTE_COLOURS:
            return "gray" if token == "grey" else token
    return next(iter(tokens), "unknown")


def normalize_object_type_label(label: str) -> str:
    s = label
    s = OBJECT_TYPE_MAP.get(s, s)
    for suf in OBJECT_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    return norm_spaces(s)


def normalize_clothing_label(label: str) -> str:
    s = label.replace("_", " ")
    if " or " in s:
        s = s.split(" or ")[0]
    if "overalls" in s or s.endswith(" overall") or s == "overall":
        return "overalls"
    s = CLOTHING_MAP.get(s, s)
    s = s.replace("short-sleeved", "short sleeved")
    for pat in SHIRT_PATTERNS:
        if re.match(pat, s):
            return "shirt"
    if s.endswith(" shirt"):
        return "shirt"
    return s


def normalize_label(cohort: str, dimension: str, raw_label: str) -> str:
    """Normalise label tokens based on cohort/dimension context."""
    label = raw_label.strip().lower()
    if not label:
        label = "unknown"
    if any(tok in label for tok in UNKNOWN_TOKENS):
        label = "unknown"

    dim = dimension
    if dim == "object_counts_by_position":
        tokens = re.findall(r"[a-z0-9]+", label)
        plane = next((t for t in tokens if t in PLANE_TOKENS), None)
        side = next((t for t in tokens if t in SIDE_TOKENS), None)
        if plane and side:
            return f"{plane} {side}"
        return label

    if cohort == "objects" and dim in {"type", "subtype"}:
        return normalize_object_type_label(label)

    if cohort == "people" and dim == "clothing_garment":
        return normalize_clothing_label(label)

    if dim.endswith("activities"):
        label = normalize_activities(label)
        return label or "unknown"

    if dim.endswith("present") or dim.endswith("presence"):
        return normalize_presence_label(label)

    if dim == "gender_presentation":
        return normalize_gender_label(label)

    if "color_temperature" in dim:
        return normalize_color_temperature_label(label)

    if "contrast_level" in dim:
        return normalize_contrast_label(label)

    tokens = normalize_label_generic(dim, label)
    tokens = [VALUE_NORMALISATION.get(t, t) for t in tokens]

    if "palette" in dim:
        return normalize_palette_label(tokens)

    return " ".join(tokens)


def normalize_label_generic(dim_slug: str, label: str) -> list[str]:
    tokens = [t for t in re.findall(r"[a-z0-9]+", label.lower())]
    dim_tokens = set(dim_slug.split("_"))
    return [t for t in tokens if t not in dim_tokens] or ["unknown"]


def normalize_activities(label: str) -> str:
    s = re.sub(r"\bactivities\b", "", label, flags=re.IGNORECASE)
    s = re.sub(r"\bactivity\b", "", s, flags=re.IGNORECASE)
    return norm_spaces(s)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def iter_source_csvs() -> Iterator[Path]:
    for path in DATASET_ROOT.rglob("aggregation_counting/*.csv"):
        if ROLE_OUTPUT_DIR in path.parents:
            continue
        if "general_attributes" in path.parts:
            continue
        yield path


def aggregate_roles() -> Dict[str, Dict[Tuple[str, str, str], int]]:
    role_totals: Dict[str, Dict[Tuple[str, str, str], int]] = defaultdict(lambda: defaultdict(int))

    for csv_path in iter_source_csvs():
        role = csv_path.parent.parent.name
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                cohort = normalize_cohort(row.get("cohort", "unknown"))
                dimension = normalize_dimension_key(row.get("dimension", "unknown"))
                label = normalize_label(cohort, dimension, row.get("label", "unknown"))
                try:
                    count = int(row.get("count", 0) or 0)
                except Exception:
                    try:
                        count = int(float(row.get("count", 0) or 0))
                    except Exception:
                        count = 0
                role_totals[role][(cohort, dimension, label)] += count
    return role_totals


def write_role_rollups(role_totals: Dict[str, Dict[Tuple[str, str, str], int]]) -> None:
    ROLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for role, counts in role_totals.items():
        out_path = ROLE_OUTPUT_DIR / f"{slugify_name(role)}.csv"
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["cohort", "dimension", "label", "count"])
            for (cohort, dimension, label), count in sorted(
                counts.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
            ):
                writer.writerow([cohort, dimension, label, int(count)])
        print(f"[Aggregated] {out_path}")


def build_general_attribute_rollup(role_totals: Dict[str, Dict[Tuple[str, str, str], int]]) -> None:
    totals: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for counts in role_totals.values():
        for (cohort, dimension, label), count in counts.items():
            key = (cohort, dimension)
            target = GENERAL_ATTRIBUTE_MAP.get(key) or GENERAL_ATTRIBUTE_MAP.get(("*", dimension))
            if not target:
                continue
            attribute, attribute_cohort = target
            totals[(attribute, attribute_cohort)][label] += count

    with GENERAL_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cohort", "dimension", "label", "count"])
        for attribute in GENERAL_ATTRIBUTES:
            for (attr_name, attr_cohort), labels in sorted(
                totals.items(), key=lambda item: (item[0][0], item[0][1])
            ):
                if attr_name != attribute:
                    continue
                for label, count in sorted(labels.items(), key=lambda item: (item[0], item[1])):
                    writer.writerow([attr_cohort, attr_name, label, int(count)])

    print(f"[Aggregated] {GENERAL_OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate role CSVs across contexts.")
    parser.add_argument(
        "--general",
        action="store_true",
        help="Also produce dataset/general_attributes_rollup.csv",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    role_totals = aggregate_roles()
    if not role_totals:
        print("[Info] No aggregation inputs found.")
        return 0
    write_role_rollups(role_totals)
    if args.general:
        build_general_attribute_rollup(role_totals)
    print(f"[Done] Wrote {len(role_totals)} role aggregates to {ROLE_OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
