"""Frequency analysis for each prompt's description files.

For every prompt directory that contains a ``descriptions/`` folder, this script
collects all per-image JSON files, extracts image-level attribute sets, and
writes frequency summaries (JSON and CSV) into ``frequency/`` located beside
``descriptions/``. Only newly discovered description JSON files are processed
on each run, making repeated invocations incremental.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

from .io_utils import write_atomic_csv, write_atomic_text

logger = logging.getLogger(__name__)

CATEGORIES: Tuple[str, ...] = (
    "scene_mood",
    "scene_color_temperature",
    "scene_contrast_level",
    "scene_aesthetic_qualities",
    "scene_dominant_colors",
    "scene_weather",
    "scene_ceiling_color",
    "scene_floor_color",
    "scene_wall_color",
    "camera_depth_of_field",
    "camera_framing",
    "camera_perspective",
    "people_gender_presentation",
    "people_skin_tone",
    "people_age_range",
    "people_body_type",
    "people_pose",
    "people_expression",
    "people_activities",
    "people_accessories",
    "people_eyewear_type",
    "people_eyewear_present",
    "people_head_covering_type",
    "people_head_covering_present",
    "people_hair_present",
    "people_hair_color",
    "people_hair_style",
    "people_facial_hair_present",
    "people_facial_hair_style",
    "people_facial_hair_color",
    "people_role_hint",
    "people_clothing_garment",
    "people_clothing_garment_color",
    "objects_items",
    "objects_size",
    "objects_color",
    "objects_material",
    "objects_item_size",
    "objects_item_color",
    "objects_item_material",
    "safety_hazards",
    "safety_nsfw",
)


def _is_token(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered not in {"", "unknown", "none"}
    if isinstance(value, (int, float)):
        return True
    return False


def _ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _default_counter(initial: Optional[Dict[str, int]] = None) -> defaultdict:
    counter: defaultdict = defaultdict(int)
    if initial:
        for token, count in initial.items():
            counter[token] = int(count)
    return counter


def _new_aggregator() -> Dict[str, Any]:
    data: Dict[str, Any] = {category: _default_counter() for category in CATEGORIES}
    data["totals"] = {"images": 0, "people_instances": 0, "object_instances": 0}
    return data


def _make_aggregator_from_serialized(serialized: Dict[str, Any]) -> Dict[str, Any]:
    agg = _new_aggregator()
    for category in CATEGORIES:
        agg[category] = _default_counter(serialized.get(category, {}))
    totals = serialized.get("totals", {})
    agg["totals"] = {
        "images": int(totals.get("images", 0)),
        "people_instances": int(totals.get("people_instances", 0)),
        "object_instances": int(totals.get("object_instances", 0)),
    }
    return agg


def _convert_counts(agg: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for category in CATEGORIES:
        counter: Dict[str, int] = agg.get(category, {})
        result[category] = dict(sorted(counter.items()))
    result["totals"] = dict(agg.get("totals", {}))
    return result


def _iter_description_dirs(dataset_root: Path) -> List[Path]:
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        logger.warning("Dataset root %s does not exist.", dataset_root)
        return []
    dirs = [p for p in dataset_root.rglob("descriptions") if p.is_dir()]
    return sorted(dirs)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None



def _extract_image_tokens(data: Dict[str, Any]) -> Tuple[Dict[str, Set[str]], int, int]:
    tokens: Dict[str, Set[str]] = {category: set() for category in CATEGORIES}

    scene = data.get("cohorts", {}).get("scene_appearance", {})
    mood = scene.get("mood")
    if _is_token(mood):
        tokens["scene_mood"].add(str(mood))
    color_temp = scene.get("color_temperature")
    if _is_token(color_temp):
        tokens["scene_color_temperature"].add(str(color_temp))
    contrast = scene.get("contrast_level")
    if _is_token(contrast):
        tokens["scene_contrast_level"].add(str(contrast))
    for quality in _ensure_list(scene.get("aesthetic_qualities")):
        if _is_token(quality):
            tokens["scene_aesthetic_qualities"].add(str(quality))
    for color in _ensure_list(scene.get("dominant_colors")):
        if _is_token(color):
            tokens["scene_dominant_colors"].add(str(color))
    weather = scene.get("weather")
    if _is_token(weather):
        tokens["scene_weather"].add(str(weather))
    for key, category in (
        ("ceiling_color", "scene_ceiling_color"),
        ("floor_color", "scene_floor_color"),
        ("wall_color", "scene_wall_color"),
    ):
        value = scene.get(key)
        if _is_token(value):
            tokens[category].add(str(value))

    camera = data.get("cohorts", {}).get("camera", {})
    depth = camera.get("depth_of_field")
    if _is_token(depth):
        tokens["camera_depth_of_field"].add(str(depth))
    framing = camera.get("framing")
    if _is_token(framing):
        tokens["camera_framing"].add(str(framing))
    perspective = camera.get("perspective")
    if _is_token(perspective):
        tokens["camera_perspective"].add(str(perspective))

    people_section = data.get("cohorts", {}).get("people", {})
    persons = _ensure_list(people_section.get("persons"))
    people_instances = 0
    for person in persons:
        if not isinstance(person, dict):
            continue
        people_instances += 1

        for key, category in (
            ("gender_presentation", "people_gender_presentation"),
            ("skin_tone", "people_skin_tone"),
            ("age_range", "people_age_range"),
            ("body_type", "people_body_type"),
            ("pose", "people_pose"),
            ("expression", "people_expression"),
            ("role_hint", "people_role_hint"),
            ("eyewear_present", "people_eyewear_present"),
            ("head_covering_present", "people_head_covering_present"),
            ("hair_present", "people_hair_present"),
            ("hair_color", "people_hair_color"),
            ("hair_style", "people_hair_style"),
            ("facial_hair_present", "people_facial_hair_present"),
            ("facial_hair_style", "people_facial_hair_style"),
            ("facial_hair_color", "people_facial_hair_color"),
        ):
            value = person.get(key)
            if isinstance(value, list):
                for subval in value:
                    if _is_token(subval):
                        tokens[category].add(str(subval))
            elif _is_token(value):
                tokens[category].add(str(value))

        for activity in _ensure_list(person.get("activities")):
            if _is_token(activity):
                tokens["people_activities"].add(str(activity))

        for accessory in _ensure_list(person.get("accessories")):
            if _is_token(accessory):
                tokens["people_accessories"].add(str(accessory))

        for eyewear in _ensure_list(person.get("eyewear_type")):
            if _is_token(eyewear):
                tokens["people_eyewear_type"].add(str(eyewear))

        for head_cover in _ensure_list(person.get("head_covering_type")):
            if _is_token(head_cover):
                tokens["people_head_covering_type"].add(str(head_cover))

        for clothing in _ensure_list(person.get("clothing")):
            if not isinstance(clothing, dict):
                continue
            garment = clothing.get("clothing_garment")
            if _is_token(garment):
                garment_str = str(garment)
                tokens["people_clothing_garment"].add(garment_str)
                color_values = sorted(
                    str(color)
                    for color in _ensure_list(clothing.get("clothing_color"))
                    if _is_token(color)
                )
                if color_values:
                    combo = f"{garment_str}|colors={','.join(color_values)}"
                    tokens["people_clothing_garment_color"].add(combo)

    objects_section = data.get("cohorts", {}).get("objects", {})
    obj_items = objects_section.get("items") or {}
    object_instances = 0
    if isinstance(obj_items, dict):
        for object_name, attrs in obj_items.items():
            name_token = str(object_name)
            if _is_token(object_name):
                tokens["objects_items"].add(name_token)
            object_instances += 1
            if not isinstance(attrs, dict):
                continue
            size = attrs.get("size")
            if _is_token(size):
                tokens["objects_size"].add(str(size))
                if _is_token(object_name):
                    tokens["objects_item_size"].add(f"{name_token}|size={size}")
            for color in _ensure_list(attrs.get("color")):
                if _is_token(color):
                    tokens["objects_color"].add(str(color))
                    if _is_token(object_name):
                        tokens["objects_item_color"].add(f"{name_token}|color={color}")
            for material in _ensure_list(attrs.get("material")):
                if _is_token(material):
                    tokens["objects_material"].add(str(material))
                    if _is_token(object_name):
                        tokens["objects_item_material"].add(f"{name_token}|material={material}")

    safety_section = data.get("cohorts", {}).get("safety", {})
    hazard = safety_section.get("hazards")
    if _is_token(hazard):
        tokens["safety_hazards"].add(str(hazard))
    nsfw = safety_section.get("nsfw")
    if _is_token(nsfw):
        tokens["safety_nsfw"].add(str(nsfw))

    return tokens, people_instances, object_instances
def _load_prompt_state(freq_dir: Path) -> Tuple[Dict[str, Any], Set[str]]:
    agg = _new_aggregator()
    processed: Set[str] = set()

    freq_json = freq_dir / "frequencies.json"
    if freq_json.is_file():
        try:
            payload = json.loads(freq_json.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                agg = _make_aggregator_from_serialized(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load existing frequencies (%s): %s", freq_json, exc)

    processed_path = freq_dir / "processed_files.json"
    if processed_path.is_file():
        try:
            entries = json.loads(processed_path.read_text(encoding="utf-8"))
            if isinstance(entries, list):
                processed = set(str(entry) for entry in entries)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load processed file list (%s): %s", processed_path, exc)

    return agg, processed


def _write_prompt_outputs(freq_dir: Path, aggregator: Dict[str, Any], processed: Set[str]) -> None:
    serialized = _convert_counts(aggregator)
    write_atomic_text(freq_dir / "frequencies.json", json.dumps(serialized, indent=2, sort_keys=True))

    rows: List[List[str]] = [["category", "token", "count"]]
    for category in CATEGORIES:
        for token, count in serialized[category].items():
            rows.append([category, token, str(count)])
    rows.extend(
        [
            ["totals", "images", str(serialized["totals"].get("images", 0))],
            ["totals", "people_instances", str(serialized["totals"].get("people_instances", 0))],
            ["totals", "object_instances", str(serialized["totals"].get("object_instances", 0))],
        ]
    )
    write_atomic_csv(freq_dir / "frequencies.csv", rows)
    write_atomic_text(freq_dir / "processed_files.json", json.dumps(sorted(processed), indent=2))


def _process_prompt(
    descriptions_dir: Path,
    dataset_root: Path,
) -> Tuple[str, int, int]:
    prompt_root = descriptions_dir.parent
    freq_dir = prompt_root / "frequency"
    freq_dir.mkdir(parents=True, exist_ok=True)

    aggregator, processed = _load_prompt_state(freq_dir)

    descriptions_dir = descriptions_dir.resolve()
    dataset_root = dataset_root.resolve()
    try:
        prompt_label = str(prompt_root.resolve().relative_to(dataset_root))
    except ValueError:
        prompt_label = str(prompt_root.resolve())

    json_files = sorted(descriptions_dir.glob("*.json"))
    new_entries: List[Tuple[Path, str]] = []
    for json_path in json_files:
        try:
            rel = str(json_path.resolve().relative_to(dataset_root))
        except ValueError:
            rel = str(json_path.resolve())
        if rel not in processed:
            new_entries.append((json_path, rel))

    if not new_entries:
        return prompt_label, 0, aggregator["totals"]["images"]

    new_count = 0
    for json_path, rel in new_entries:
        data = _load_json(json_path)
        if not isinstance(data, dict):
            continue
        tokens, people_instances, object_instances = _extract_image_tokens(data)
        aggregator["totals"]["images"] += 1
        aggregator["totals"]["people_instances"] += people_instances
        aggregator["totals"]["object_instances"] += object_instances
        for category in CATEGORIES:
            for token in tokens.get(category, set()):
                aggregator[category][token] += 1
        processed.add(rel)
        new_count += 1

    _write_prompt_outputs(freq_dir, aggregator, processed)
    return prompt_label, new_count, aggregator["totals"]["images"]


class FrequencyCounterResult(NamedTuple):
    prompts_visited: int
    prompts_updated: int
    new_images: int
    total_images: int


def compute_frequencies(dataset_root: Path) -> FrequencyCounterResult:
    dataset_root = dataset_root.resolve()
    description_dirs = _iter_description_dirs(dataset_root)

    if not description_dirs:
        logger.info("No description directories found under %s. Nothing to do.", dataset_root)
        return FrequencyCounterResult(0, 0, 0, 0)

    total_prompts = len(description_dirs)
    total_new_images = 0
    total_images_after = 0
    prompts_with_updates = 0

    for desc_dir in description_dirs:
        prompt_label, new_images, total_images = _process_prompt(desc_dir, dataset_root)
        total_images_after += total_images
        if new_images:
            prompts_with_updates += 1
            total_new_images += new_images
            logger.info("Processed %d new images for %s", new_images, prompt_label)
        else:
            logger.debug("No new images for %s", prompt_label)

    return FrequencyCounterResult(
        prompts_visited=total_prompts,
        prompts_updated=prompts_with_updates,
        new_images=total_new_images,
        total_images=total_images_after,
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute image-level frequency statistics for each prompt."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("./dataset"),
        help="Path to the dataset root (default: ./dataset)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    result = compute_frequencies(args.dataset)

    dataset_root = args.dataset.resolve()
    if result.prompts_visited == 0:
        print(f"No description directories found under {dataset_root}. Nothing to do.")
        return

    print(
        f"Visited {result.prompts_visited} prompt folders. "
        f"Processed {result.new_images} new images across {result.prompts_updated} prompt(s). "
        f"Totals now cover {result.total_images} images across the dataset. "
        f"Frequency outputs are stored in each prompt's frequency/ directory."
    )


if __name__ == "__main__":
    main()
