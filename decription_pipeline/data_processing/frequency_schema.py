"""Cohort/dimension schema definitions and CSV helpers for frequency counting."""

from collections import OrderedDict
from typing import Dict, Iterable, Iterator, Tuple

# Ordered cohort -> dimension -> category mapping
COHORT_DIMENSIONS: "OrderedDict[str, OrderedDict[str, str]]" = OrderedDict(
    [
        (
            "scene_appearance",
            OrderedDict(
                [
                    ("mood", "scene_mood"),
                    ("color_temperature", "scene_color_temperature"),
                    ("contrast_level", "scene_contrast_level"),
                    ("aesthetic_qualities", "scene_aesthetic_qualities"),
                    ("dominant_colors", "scene_dominant_colors"),
                    ("weather", "scene_weather"),
                ]
            ),
        ),
        (
            "camera",
            OrderedDict(
                [
                    ("depth_of_field", "camera_depth_of_field"),
                    ("framing", "camera_framing"),
                    ("perspective", "camera_perspective"),
                ]
            ),
        ),
        (
            "people",
            OrderedDict(
                [
                    ("gender_presentation", "people_gender_presentation"),
                    ("skin_tone", "people_skin_tone"),
                    ("age_range", "people_age_range"),
                    ("body_type", "people_body_type"),
                    ("pose", "people_pose"),
                    ("expression", "people_expression"),
                    ("activities", "people_activities"),
                    ("accessories", "people_accessories"),
                    ("eyewear_type", "people_eyewear_type"),
                    ("eyewear_present", "people_eyewear_present"),
                    ("head_covering_type", "people_head_covering_type"),
                    ("head_covering_present", "people_head_covering_present"),
                    ("hair_present", "people_hair_present"),
                    ("hair_color", "people_hair_color"),
                    ("hair_style", "people_hair_style"),
                    ("tattoo_present", "people_tattoo_present"),
                    ("facial_hair_present", "people_facial_hair_present"),
                    ("facial_hair_style", "people_facial_hair_style"),
                    ("facial_hair_color", "people_facial_hair_color"),
                    ("clothing_garment", "people_clothing_garment"),
                ]
            ),
        ),
        (
            "objects",
            OrderedDict(
                [
                    ("items", "objects_items"),
                ]
            ),
        ),
        (
            "totals",
            OrderedDict(
                [
                    ("images", None),
                    ("people_instances", None),
                    ("object_instances", None),
                ]
            ),
        ),
    ]
)

CATEGORY_TO_COHORT_DIMENSION: Dict[str, Tuple[str, str]] = {
    category: (cohort, dimension)
    for cohort, dimensions in COHORT_DIMENSIONS.items()
    for dimension, category in dimensions.items()
    if category is not None
}


def create_counts_structure() -> "OrderedDict[str, OrderedDict[str, Dict[str, int]]]":
    """Initialise an ordered cohort/dimension mapping with empty label dicts."""
    structure: "OrderedDict[str, OrderedDict[str, Dict[str, int]]]" = OrderedDict()
    for cohort, dimensions in COHORT_DIMENSIONS.items():
        structure[cohort] = OrderedDict((dimension, {}) for dimension in dimensions)
    return structure


def normalise_cohort_structure(raw: object) -> "OrderedDict[str, OrderedDict[str, Dict[str, int]]]":
    """Coerce arbitrary JSON data into the standard cohort structure."""
    structure = create_counts_structure()
    if not isinstance(raw, dict):
        return structure

    legacy_keys = [key for key in raw.keys() if key in CATEGORY_TO_COHORT_DIMENSION]
    if legacy_keys:
        for category in legacy_keys:
            labels = raw.get(category)
            mapping = CATEGORY_TO_COHORT_DIMENSION.get(category)
            if mapping is None or not isinstance(labels, dict):
                continue
            cohort, dimension = mapping
            target_labels = structure.setdefault(cohort, OrderedDict()).setdefault(dimension, {})
            for label, value in labels.items():
                target_labels[str(label)] = int(value)

        totals_payload = raw.get("totals")
        if isinstance(totals_payload, dict):
            totals_block = structure.setdefault("totals", OrderedDict())
            for metric, value in totals_payload.items():
                metric_dict = totals_block.setdefault(metric, {})
                if isinstance(value, dict):
                    for label, inner in value.items():
                        metric_dict[str(label)] = int(inner)
                else:
                    metric_dict[str(metric)] = int(value)
        return structure

    for cohort, dimensions in raw.items():
        if not isinstance(cohort, str) or not isinstance(dimensions, dict):
            continue
        target_dims = structure.setdefault(cohort, OrderedDict())
        for dimension, labels in dimensions.items():
            if not isinstance(dimension, str):
                continue
            target_labels = target_dims.setdefault(dimension, {})
            if isinstance(labels, dict):
                for label, value in labels.items():
                    target_labels[str(label)] = int(value)
            elif isinstance(labels, (int, float)):
                target_labels[str(dimension)] = int(labels)
    return structure


def iter_dimension_labels(
    cohorts: "OrderedDict[str, OrderedDict[str, Dict[str, int]]]"
) -> Iterator[Tuple[str, str, Dict[str, int]]]:
    """Iterate over (cohort, dimension, labels dict) triples."""
    for cohort, dimensions in cohorts.items():
        for dimension, labels in dimensions.items():
            yield cohort, dimension, labels


def build_csv_rows(
    cohorts: "OrderedDict[str, OrderedDict[str, Dict[str, int]]]"
) -> list[list[str]]:
    """Create CSV rows with columns: cohort, dimension, label, count, bin."""
    rows: list[list[str]] = [["cohort", "dimension", "label", "count", "bin"]]
    for cohort, dimension, labels in iter_dimension_labels(cohorts):
        if not isinstance(labels, dict) or not labels:
            continue
        bin_size = len(labels)
        for label, count in sorted(labels.items()):
            rows.append([cohort, dimension, label, str(int(count)), str(bin_size)])
    return rows


__all__ = [
    "COHORT_DIMENSIONS",
    "CATEGORY_TO_COHORT_DIMENSION",
    "create_counts_structure",
    "normalise_cohort_structure",
    "iter_dimension_labels",
    "build_csv_rows",
]
