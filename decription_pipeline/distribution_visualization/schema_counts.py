
"""Tools for aggregating Stage 2 structured description outputs into frequency counts."""
import json
import logging
import re
import unicodedata
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

warnings.filterwarnings("ignore", message=".*force_all_finite.*")

import numpy as np
import spacy
from rapidfuzz import fuzz as rf_fuzz
from rapidfuzz import process as rf_process
from sentence_transformers import SentenceTransformer
import hdbscan  # type: ignore
import matplotlib.colors as mcolors

CSS_COLOR_SET = sorted({c.lower() for c in mcolors.cnames.keys()})

LOGGER = logging.getLogger(__name__)

@dataclass
class FrequencyCounterConfig:
    model_name: str = "Alibaba-NLP/gte-large-en-v1.5"
    trust_remote_code: bool = True
    hdbscan_min_cluster_size: int = 4
    hdbscan_min_samples: Optional[int] = None
    enum_embed_threshold: float = 0.62
    css_color_sim_threshold: float = 0.48


_NLP = None
_MODEL = None
_CSS_EMB = None
_ENUM_VEC_CACHE: Dict[str, np.ndarray] = {}

STOPWORDS = set(
    "a an and the of for to in on at from with this that is are was were be been as it its their his her your our".split()
)


def _load_spacy():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_sm")
        except Exception:
            _NLP = spacy.blank("en")
    return _NLP


def _get_encoder(cfg: FrequencyCounterConfig):
    global _MODEL
    if _MODEL is None:
        LOGGER.info("Loading SentenceTransformer: %s", cfg.model_name)
        _MODEL = SentenceTransformer(cfg.model_name, trust_remote_code=cfg.trust_remote_code)
    return _MODEL


def _get_css_embeddings(cfg: FrequencyCounterConfig):
    global _CSS_EMB
    if _CSS_EMB is None:
        _CSS_EMB = _get_encoder(cfg).encode(
            CSS_COLOR_SET, show_progress_bar=False, normalize_embeddings=True
        )
    return _CSS_EMB


def strip_accents(value: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", str(value)) if not unicodedata.combining(ch)
    )


def basic_clean(value: str) -> str:
    base = strip_accents(str(value)).lower()
    base = re.sub(r"[^\w\s\-]", " ", base)
    base = base.replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", base).strip()


def lemma_tokens(value: str) -> List[str]:
    nlp = _load_spacy()
    text_value = basic_clean(value)
    if not text_value:
        return []
    if not hasattr(nlp, "pipe"):
        tokens: List[str] = []
        for token in text_value.split():
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                tokens.append(token[:-1])
            else:
                tokens.append(token)
        return tokens
    doc = nlp(text_value)
    normalized: List[str] = []
    for word in doc:
        if not (word.is_alpha or word.is_digit):
            continue
        lemma = (word.lemma_ or word.text).lower()
        if lemma in STOPWORDS:
            continue
        normalized.append(lemma)
    return normalized


def norm_text(value: str) -> str:
    return " ".join(lemma_tokens(value))


def to_snake(value: str) -> str:
    clean_value = basic_clean(value)
    return "_".join(token for token in clean_value.split() if token)

ENUM_OPTIONS: Dict[str, List[str]] = {
    "people.demographics.gender_presentation": ["male", "female", "nonbinary", "unknown"],
    "people.demographics.skin_tone": ["light", "medium", "dark", "unknown"],
    "people.demographics.age_range": [
        "child",
        "teen",
        "young_adult",
        "middle_aged",
        "older_adult",
        "unknown",
    ],
    "people.appearance.hair.present": ["yes", "no", "unknown"],
    "people.appearance.facial_hair.present": ["yes", "no", "unknown"],
    "people.appearance.eyewear.present": ["yes", "no", "unknown"],
    "people.appearance.head_covering.present": ["yes", "no", "unknown"],
    "people.pose_activity.gaze_direction": ["left", "center", "right", "up", "down", "unknown"],
    "people.pose_activity.orientation": ["front", "three_quarter", "profile", "back", "unknown"],
    "people.pose_activity.occlusions": ["none", "partial", "significant", "unknown"],
    "environment.indoor_outdoor": ["indoor", "outdoor", "unknown"],
    "camera.angle": ["high", "eye_level", "low", "unknown"],
    "camera.focal_length": ["wide", "normal", "short_tele", "tele", "unknown"],
    "camera.depth_of_field": ["shallow", "moderate", "deep", "unknown"],
    "lighting.color_temperature": ["warm", "neutral", "cool", "unknown"],
    "lighting.contrast_level": ["low", "medium", "high", "unknown"],
    "lighting.saturation_level": ["muted", "neutral", "vivid", "unknown"],
    "objects.plane": ["foreground", "midground", "background", "unknown"],
    "objects.side": ["left", "center", "right", "unknown"],
    "people.positions.plane": ["foreground", "midground", "background", "unknown"],
    "people.positions.side": ["left", "center", "right", "unknown"],
}


def _enum_vecs(field: str, cfg: FrequencyCounterConfig) -> np.ndarray:
    if field not in _ENUM_VEC_CACHE:
        _ENUM_VEC_CACHE[field] = _get_encoder(cfg).encode(
            ENUM_OPTIONS[field], show_progress_bar=False, normalize_embeddings=True
        )
    return _ENUM_VEC_CACHE[field]


def canonicalize_enum(field: str, raw_value: str, cfg: FrequencyCounterConfig) -> str:
    normalized_value = basic_clean(raw_value)
    if not normalized_value:
        return "unknown" if "unknown" in ENUM_OPTIONS.get(field, []) else ""
    options = ENUM_OPTIONS.get(field, [])
    if not options:
        return normalized_value
    best = rf_process.extractOne(normalized_value, options, scorer=rf_fuzz.token_set_ratio)
    if best and best[1] >= 96:
        return best[0]
    vectors = _enum_vecs(field, cfg)
    value_vec = _get_encoder(cfg).encode(
        [normalized_value], show_progress_bar=False, normalize_embeddings=True
    )[0]
    sims = vectors @ value_vec
    idx = int(np.argmax(sims))
    if float(sims[idx]) >= cfg.enum_embed_threshold:
        return options[idx]
    return normalized_value


def project_color(value: str, cfg: FrequencyCounterConfig) -> Optional[str]:
    if not CSS_COLOR_SET:
        return None
    vector = _get_encoder(cfg).encode([value], show_progress_bar=False, normalize_embeddings=True)[0]
    css_vectors = _get_css_embeddings(cfg)
    sims = css_vectors @ vector
    idx = int(np.argmax(sims))
    if float(sims[idx]) >= cfg.css_color_sim_threshold:
        return CSS_COLOR_SET[idx]
    best = rf_process.extractOne(value, CSS_COLOR_SET, scorer=rf_fuzz.token_set_ratio)
    if best and best[1] >= 92:
        return best[0]
    return None


def normalize_color_token(token: str, cfg: FrequencyCounterConfig) -> str:
    projected = project_color(token, cfg)
    return to_snake(projected if projected else token)

def _iter_clean(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if item not in (None, "") and str(item).strip()]
    string_value = str(value).strip()
    return [string_value] if string_value else []


def inc_vc(node: Dict[str, Any], key: str, inc: int = 1) -> None:
    value_counts = node.setdefault("value_counts", {})
    value_counts[key] = int(value_counts.get(key, 0)) + inc


def ensure_position_bins(root: Dict[str, Any]) -> None:
    by_position = root["objects"].setdefault("by_position", {})
    background_counts = root["background"].setdefault("object_counts_by_position", {})
    for plane in ("foreground", "midground", "background"):
        for side in ("left", "center", "right"):
            key = f"{plane}.{side}"
            by_position.setdefault(key, {"type_counts": {}, "type_subtype_counts": {}})
            background_counts.setdefault(key, {"total": 0})


def ensure_type_bucket(root: Dict[str, Any], type_key: str) -> Dict[str, Any]:
    by_type = root["objects"].setdefault("by_type", {})
    if type_key not in by_type:
        by_type[type_key] = {
            "count": 0,
            "by_subtype": {},
            "positions": {"plane_side": {"value_counts": {}}},
            "attributes": {
                "colors": {"value_counts": {}},
                "material": {"value_counts": {}},
                "texture": {"value_counts": {}},
                "finish": {"value_counts": {}},
                "condition": {"value_counts": {}},
                "state": {"value_counts": {}},
                "size_class": {"value_counts": {}},
            },
            "quantity": {
                "exact_sum": 0,
                "exact_present_count": 0,
                "approx_tokens": {"value_counts": {}},
            },
        }
    return by_type[type_key]


def ensure_type_subtype_bucket(root: Dict[str, Any], type_key: str, sub_key: str) -> Dict[str, Any]:
    composite = f"{type_key}|||{sub_key}"
    by_pair = root["objects"].setdefault("by_type_subtype", {})
    if composite not in by_pair:
        by_pair[composite] = {
            "count": 0,
            "positions": {"plane_side": {"value_counts": {}}},
            "attributes": {
                "colors": {"value_counts": {}},
                "material": {"value_counts": {}},
                "texture": {"value_counts": {}},
                "finish": {"value_counts": {}},
                "condition": {"value_counts": {}},
                "state": {"value_counts": {}},
                "size_class": {"value_counts": {}},
            },
            "quantity": {
                "exact_sum": 0,
                "exact_present_count": 0,
                "approx_tokens": {"value_counts": {}},
            },
        }
    return by_pair[composite]


def ensure_clothing_bucket(root: Dict[str, Any], garment_type: str) -> Dict[str, Any]:
    by_type = root["people"]["clothing"].setdefault("by_garment_type", {})
    if garment_type not in by_type:
        by_type[garment_type] = {
            "count": 0,
            "color": {"value_counts": {}},
            "material": {"value_counts": {}},
            "texture": {"value_counts": {}},
            "fit_style": {"value_counts": {}},
            "pattern": {"value_counts": {}},
            "condition": {"value_counts": {}},
        }
    return by_type[garment_type]
def make_output_skeleton(num_images: int) -> Dict[str, Any]:
    import datetime as dt

    output = {
        "meta": {
            "version": "1.1",
            "generated_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "input_schema_version": "v2025-09",
            "num_images": num_images,
            "notes": "",
        },
        "totals": {
            "images": num_images,
            "people_instances": 0,
            "object_instances": 0,
            "text_instances": 0,
        },
        "atmosphere": {
            "mood": {"value_counts": {}},
            "dominant_palette": {"value_counts": {}},
            "lighting_profile": {
                "color_temperature": {"value_counts": {}},
                "contrast_level": {"value_counts": {}},
                "saturation_level": {"value_counts": {}},
                "aesthetic_qualities": {"value_counts": {}},
            },
        },
        "environment": {
            "location_type": {"value_counts": {}},
            "indoor_outdoor": {"value_counts": {}},
            "time_of_day_hint": {"value_counts": {}},
            "weather": {"value_counts": {}},
            "spatial_layout": {
                "depth": {"value_counts": {}},
                "openness": {"value_counts": {}},
                "aisle_width": {"value_counts": {}},
            },
            "surfaces": {
                "walls": {
                    "material": {"value_counts": {}},
                    "texture": {"value_counts": {}},
                    "color": {"value_counts": {}},
                    "finish": {"value_counts": {}},
                    "condition": {"value_counts": {}},
                },
                "floor": {
                    "material": {"value_counts": {}},
                    "texture": {"value_counts": {}},
                    "color": {"value_counts": {}},
                    "finish": {"value_counts": {}},
                    "condition": {"value_counts": {}},
                },
                "ceiling": {
                    "material": {"value_counts": {}},
                    "texture": {"value_counts": {}},
                    "color": {"value_counts": {}},
                    "finish": {"value_counts": {}},
                    "condition": {"value_counts": {}},
                },
            },
        },
        "people": {
            "demographics": {
                "gender_presentation": {"value_counts": {}},
                "skin_tone": {"value_counts": {}},
                "age_range": {"value_counts": {}},
                "role_hint": {"value_counts": {}},
            },
            "appearance": {
                "hair.present": {"value_counts": {}},
                "hair.style": {"value_counts": {}},
                "hair.color": {"value_counts": {}},
                "facial_hair.present": {"value_counts": {}},
                "facial_hair.style": {"value_counts": {}},
                "facial_hair.color": {"value_counts": {}},
                "eyewear.present": {"value_counts": {}},
                "eyewear.type": {"value_counts": {}},
                "eyewear.frame_color": {"value_counts": {}},
                "head_covering.present": {"value_counts": {}},
                "head_covering.type": {"value_counts": {}},
                "head_covering.color": {"value_counts": {}},
            },
            "pose_activity": {
                "pose": {"value_counts": {}},
                "activities": {"value_counts": {}},
                "gaze_direction": {"value_counts": {}},
                "orientation": {"value_counts": {}},
                "occlusions": {"value_counts": {}},
            },
            "clothing": {"by_garment_type": {}},
            "positions": {
                "plane": {"value_counts": {}},
                "side": {"value_counts": {}},
            },
        },
        "objects": {
            "type": {"value_counts": {}},
            "subtype": {"value_counts": {}},
            "plane": {"value_counts": {}},
            "side": {"value_counts": {}},
            "by_type": {},
            "by_type_subtype": {},
            "by_position": {},
        },
        "background": {"object_counts_by_position": {}},
        "texts": {
            "content": {"value_counts": {}},
            "font_style": {"value_counts": {}},
            "plane": {"value_counts": {}},
            "side": {"value_counts": {}},
            "legibility": {"value_counts": {}},
        },
        "camera": {
            "angle": {"value_counts": {}},
            "perspective": {"value_counts": {}},
            "focal_length": {"value_counts": {}},
            "depth_of_field": {"value_counts": {}},
            "framing": {"value_counts": {}},
            "crop": {"value_counts": {}},
        },
        "lighting": {
            "color_temperature": {"value_counts": {}},
            "contrast_level": {"value_counts": {}},
            "saturation_level": {"value_counts": {}},
            "sources": {
                "type": {"value_counts": {}},
                "count": {"value_counts": {}},
                "directionality": {"value_counts": {}},
                "hardness": {"value_counts": {}},
            },
            "shadows": {"value_counts": {}},
            "artifacts": {"value_counts": {}},
        },
        "safety": {
            "hazards": {"value_counts": {}},
            "nsfw": {"value_counts": {}},
        },
        "uncertainty": {"value_counts": {}},
        "dynamic_field_counts": [],
    }
    ensure_position_bins(output)
    return output
def collect_raw_for_clustering(docs: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    raw: Dict[str, List[str]] = defaultdict(list)

    def add(field: str, value: str) -> None:
        if not value:
            return
        raw[field].append(norm_text(value))

    for doc in docs:
        scene = doc.get("scene") or {}

        atmosphere = scene.get("atmosphere") or {}
        for val in _iter_clean(atmosphere.get("mood")):
            add("atmosphere.mood", val)
        for val in _iter_clean(atmosphere.get("dominant_palette")):
            add("atmosphere.dominant_palette", val)
        lighting_profile = atmosphere.get("lighting_profile") or {}
        for key in ("color_temperature", "contrast_level", "saturation_level"):
            profile_val = lighting_profile.get(key)
            if profile_val:
                add(f"atmosphere.lighting_profile.{key}", profile_val)
        for val in _iter_clean(lighting_profile.get("aesthetic_qualities")):
            add("atmosphere.lighting_profile.aesthetic_qualities", val)

        environment = scene.get("environment") or {}
        for key in ("location_type", "time_of_day_hint", "weather"):
            env_value = environment.get(key)
            if env_value:
                add(f"environment.{key}", env_value)
        spatial = environment.get("spatial_layout") or {}
        for key in ("depth", "openness", "aisle_width"):
            spatial_value = spatial.get(key)
            if spatial_value:
                add(f"environment.spatial_layout.{key}", spatial_value)
        surfaces = environment.get("surfaces") or {}
        for surface in ("walls", "floor", "ceiling"):
            for item in surfaces.get(surface) or []:
                for key in ("material", "texture", "color", "finish", "condition"):
                    values = item.get(key)
                    if values is None:
                        continue
                    for val in _iter_clean(values):
                        add(f"environment.surfaces.{surface}.{key}", val)
        for person in (scene.get("people") or {}).get("persons", []):
            if person.get("role_hint"):
                add("people.demographics.role_hint", person["role_hint"])
            for sub, style_key, color_key in [
                ("hair", "style", "color"),
                ("facial_hair", "style", "color"),
                ("eyewear", "type", "frame_color"),
                ("head_covering", "type", "color"),
            ]:
                node = person.get(sub) or {}
                for val in _iter_clean(node.get(style_key) or node.get("type")):
                    field = (
                        f"people.appearance.{sub}.style"
                        if sub in ("hair", "facial_hair")
                        else f"people.appearance.{sub}.type"
                    )
                    add(field, val)
                for val in _iter_clean(node.get(color_key) or node.get("color")):
                    color_field = (
                        f"people.appearance.{sub}.frame_color"
                        if sub == "eyewear"
                        else f"people.appearance.{sub}.color"
                    )
                    add(color_field, val)
            for garment in person.get("clothing") or []:
                if garment.get("garment_type"):
                    add("people.clothing.garment_type", garment["garment_type"])
                for key in ("color", "material", "texture", "fit_style", "pattern", "condition"):
                    for val in _iter_clean(garment.get(key)):
                        add(f"people.clothing.{key}", val)
            for val in _iter_clean(person.get("pose")):
                add("people.pose_activity.pose", val)
            for val in _iter_clean(person.get("activities")):
                add("people.pose_activity.activities", val)
            for key in ("gaze_direction", "orientation", "occlusions"):
                if person.get(key):
                    add(f"people.pose_activity.{key}", person[key])
        for obj in scene.get("objects") or []:
            if obj.get("type"):
                add("objects.type", obj["type"])
            if obj.get("subtype"):
                add("objects.subtype", obj["subtype"])
            if obj.get("plane"):
                add("objects.plane", obj["plane"])
            if obj.get("side"):
                add("objects.side", obj["side"])
            attributes = obj.get("attributes") or {}
            for key in ("colors", "material", "texture", "finish", "condition", "state", "size_class"):
                values = attributes.get(key)
                if values is None:
                    continue
                for val in _iter_clean(values):
                    add(f"objects.attributes.{key}", val)
            quantity = obj.get("quantity") or {}
            if quantity.get("approx"):
                add("objects.quantity.approx_tokens", quantity["approx"])
        for text_item in scene.get("texts") or []:
            if text_item.get("content"):
                add("texts.content", text_item["content"])
            for val in _iter_clean(text_item.get("font_style")):
                add("texts.font_style", val)
            for key in ("plane", "side", "legibility"):
                if text_item.get(key):
                    add(f"texts.{key}", text_item[key])

        camera = scene.get("camera") or {}
        for key in ("angle", "perspective", "focal_length", "depth_of_field"):
            if camera.get(key):
                add(f"camera.{key}", camera[key])
        for val in _iter_clean(camera.get("framing")):
            add("camera.framing", val)
        for val in _iter_clean(camera.get("crop")):
            add("camera.crop", val)

        lighting = scene.get("lighting") or {}
        for key in ("color_temperature", "contrast_level", "saturation_level"):
            if lighting.get(key):
                add(f"lighting.{key}", lighting[key])
        for source in lighting.get("sources") or []:
            if source.get("type"):
                add("lighting.sources.type", source["type"])
            if source.get("count"):
                add("lighting.sources.count", source["count"])
            for val in _iter_clean(source.get("directionality")):
                add("lighting.sources.directionality", val)
            if source.get("hardness"):
                add("lighting.sources.hardness", source["hardness"])
        for val in _iter_clean(lighting.get("shadows")):
            add("lighting.shadows", val)
        for val in _iter_clean(lighting.get("artifacts")):
            add("lighting.artifacts", val)

        safety = scene.get("safety") or {}
        for val in _iter_clean(safety.get("hazards")):
            add("safety.hazards", val)
        for val in _iter_clean(safety.get("nsfw")):
            add("safety.nsfw", val)
        for val in _iter_clean(scene.get("uncertainty")):
            add("uncertainty", val)

    return raw
def cluster_strings(values: List[str], cfg: FrequencyCounterConfig) -> Dict[str, str]:
    unique_values = sorted({value for value in values if value})
    if not unique_values:
        return {}
    if len(unique_values) == 1:
        return {unique_values[0]: unique_values[0]}
    matrix = _get_encoder(cfg).encode(unique_values, show_progress_bar=False, normalize_embeddings=True)
    min_cluster_size = max(2, min(cfg.hdbscan_min_cluster_size, len(unique_values)))
    min_samples = cfg.hdbscan_min_samples if cfg.hdbscan_min_samples is not None else min_cluster_size
    min_samples = max(1, min(min_samples, len(unique_values)))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(matrix)
    sims = matrix @ matrix.T
    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[int(label)].append(idx)
    output: Dict[str, str] = {}
    for label, indexes in clusters.items():
        if label == -1:
            for idx in indexes:
                output[unique_values[idx]] = unique_values[idx]
            continue
        best_idx, best_score = None, -1.0
        for idx in indexes:
            score = float(np.mean(sims[idx, indexes]))
            if score > best_score:
                best_score = score
                best_idx = idx
        medoid = unique_values[best_idx] if best_idx is not None else unique_values[indexes[0]]
        for idx in indexes:
            output[unique_values[idx]] = medoid
    return output


nCOLOR_FIELDS = {
    "atmosphere.dominant_palette",
    "environment.surfaces.walls.color",
    "environment.surfaces.floor.color",
    "environment.surfaces.ceiling.color",
    "people.appearance.hair.color",
    "people.appearance.facial_hair.color",
    "people.appearance.eyewear.frame_color",
    "people.appearance.head_covering.color",
    "people.clothing.color",
    "objects.attributes.colors",
}


def build_cluster_maps(raw: Dict[str, List[str]], cfg: FrequencyCounterConfig) -> Dict[str, Dict[str, str]]:
    maps: Dict[str, Dict[str, str]] = {}
    for field, values in raw.items():
        if field in nCOLOR_FIELDS or field in ENUM_OPTIONS:
            continue
        maps[field] = cluster_strings(values, cfg)
    return maps
def _load_documents(input_dir: Path) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    for file in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            data["_image_id"] = data.get("image_id") or file.stem
            documents.append(data)
        except Exception as exc:
            LOGGER.warning("Skipping %s: %s", file.name, exc)
    return documents


def _cluster_or_self(field: str, value: str, cmap: Dict[str, Dict[str, str]]) -> str:
    key = norm_text(value)
    token = cmap.get(field, {}).get(key, key)
    return to_snake(token)


def _enum_or_self(field: str, value: str, cfg: FrequencyCounterConfig) -> str:
    if field in ENUM_OPTIONS:
        return canonicalize_enum(field, value, cfg)
    return to_snake(norm_text(value))
def generate_counts(input_dir: Path, cfg: Optional[FrequencyCounterConfig] = None) -> Dict[str, Any]:
    config = cfg or FrequencyCounterConfig()
    docs = _load_documents(input_dir)
    if not docs:
        raise RuntimeError(f"No JSON files found under {input_dir}")

    raw = collect_raw_for_clustering(docs)
    cmap = build_cluster_maps(raw, config)
    output = make_output_skeleton(len(docs))

    def cluster(field: str, value: str) -> str:
        return _cluster_or_self(field, value, cmap)

    def enum(field: str, value: str) -> str:
        return _enum_or_self(field, value, config)

    ensure_position_bins(output)

    for doc in docs:
        scene = doc.get("scene") or {}

        atmosphere = scene.get("atmosphere") or {}
        for value in _iter_clean(atmosphere.get("mood")):
            inc_vc(output["atmosphere"]["mood"], cluster("atmosphere.mood", value))
        for value in _iter_clean(atmosphere.get("dominant_palette")):
            inc_vc(output["atmosphere"]["dominant_palette"], normalize_color_token(value, config))
        lighting_profile = atmosphere.get("lighting_profile") or {}
        for key in ("color_temperature", "contrast_level", "saturation_level"):
            value = lighting_profile.get(key)
            if value:
                inc_vc(output["atmosphere"]["lighting_profile"][key], enum(f"lighting.{key}", value))
        for value in _iter_clean(lighting_profile.get("aesthetic_qualities")):
            inc_vc(
                output["atmosphere"]["lighting_profile"]["aesthetic_qualities"],
                cluster("atmosphere.lighting_profile.aesthetic_qualities", value),
            )
        environment = scene.get("environment") or {}
        if environment.get("location_type"):
            inc_vc(
                output["environment"]["location_type"],
                cluster("environment.location_type", environment["location_type"]),
            )
        if environment.get("indoor_outdoor"):
            inc_vc(
                output["environment"]["indoor_outdoor"],
                enum("environment.indoor_outdoor", environment["indoor_outdoor"]),
            )
        if environment.get("time_of_day_hint"):
            inc_vc(
                output["environment"]["time_of_day_hint"],
                cluster("environment.time_of_day_hint", environment["time_of_day_hint"]),
            )
        if environment.get("weather"):
            inc_vc(
                output["environment"]["weather"],
                cluster("environment.weather", environment["weather"]),
            )
        spatial = environment.get("spatial_layout") or {}
        for key in ("depth", "openness", "aisle_width"):
            value = spatial.get(key)
            if value:
                inc_vc(
                    output["environment"]["spatial_layout"][key],
                    cluster(f"environment.spatial_layout.{key}", value),
                )
        surfaces = environment.get("surfaces") or {}
        for surface in ("walls", "floor", "ceiling"):
            for item in surfaces.get(surface) or []:
                for key in ("material", "texture", "color", "finish", "condition"):
                    values = item.get(key)
                    if values is None:
                        continue
                    for value in _iter_clean(values):
                        if key == "color":
                            inc_vc(
                                output["environment"]["surfaces"][surface]["color"],
                                normalize_color_token(value, config),
                            )
                        else:
                            inc_vc(
                                output["environment"]["surfaces"][surface][key],
                                cluster(f"environment.surfaces.{surface}.{key}", value),
                            )
        people = (scene.get("people") or {}).get("persons", [])
        output["totals"]["people_instances"] += len(people)
        for person in people:
            if person.get("gender_presentation"):
                inc_vc(
                    output["people"]["demographics"]["gender_presentation"],
                    enum("people.demographics.gender_presentation", person["gender_presentation"]),
                )
            if person.get("skin_tone"):
                inc_vc(
                    output["people"]["demographics"]["skin_tone"],
                    enum("people.demographics.skin_tone", person["skin_tone"]),
                )
            if person.get("age_range"):
                inc_vc(
                    output["people"]["demographics"]["age_range"],
                    enum("people.demographics.age_range", person["age_range"]),
                )
            if person.get("role_hint"):
                inc_vc(
                    output["people"]["demographics"]["role_hint"],
                    cluster("people.demographics.role_hint", person["role_hint"]),
                )

            def handle_appearance(sub: str, style_key: str, color_key: str, present_key: str = "present") -> None:
                node = person.get(sub) or {}
                if node.get(present_key) is not None:
                    inc_vc(
                        output["people"]["appearance"][f"{sub}.present"],
                        enum(f"people.appearance.{sub}.present", node[present_key]),
                    )
                for value in _iter_clean(node.get(style_key) or node.get("type")):
                    field = (
                        f"people.appearance.{sub}.style"
                        if sub in ("hair", "facial_hair")
                        else f"people.appearance.{sub}.type"
                    )
                    inc_vc(
                        output["people"]["appearance"][field.split("people.appearance.", 1)[1]],
                        cluster(field, value),
                    )
                for value in _iter_clean(node.get(color_key) or node.get("color")):
                    color_field = f"{sub}.{color_key if sub == 'eyewear' else 'color'}"
                    inc_vc(
                        output["people"]["appearance"][color_field],
                        normalize_color_token(value, config),
                    )

            handle_appearance("hair", "style", "color")
            handle_appearance("facial_hair", "style", "color")
            handle_appearance("eyewear", "type", "frame_color")
            handle_appearance("head_covering", "type", "color")

            for garment in person.get("clothing") or []:
                if garment.get("garment_type"):
                    garment_key = cluster("people.clothing.garment_type", garment["garment_type"])
                else:
                    garment_key = "unknown"
                bucket = ensure_clothing_bucket(output, garment_key)
                bucket["count"] += 1
                for key in ("color", "material", "texture", "fit_style", "pattern", "condition"):
                    for value in _iter_clean(garment.get(key)):
                        if key == "color":
                            inc_vc(bucket["color"], normalize_color_token(value, config))
                        else:
                            inc_vc(bucket[key], cluster(f"people.clothing.{key}", value))

            for value in _iter_clean(person.get("pose")):
                inc_vc(output["people"]["pose_activity"]["pose"], cluster("people.pose_activity.pose", value))
            for value in _iter_clean(person.get("activities")):
                inc_vc(
                    output["people"]["pose_activity"]["activities"],
                    cluster("people.pose_activity.activities", value),
                )
            if person.get("gaze_direction"):
                inc_vc(
                    output["people"]["pose_activity"]["gaze_direction"],
                    enum("people.pose_activity.gaze_direction", person["gaze_direction"]),
                )
            if person.get("orientation"):
                inc_vc(
                    output["people"]["pose_activity"]["orientation"],
                    enum("people.pose_activity.orientation", person["orientation"]),
                )
            if person.get("occlusions"):
                inc_vc(
                    output["people"]["pose_activity"]["occlusions"],
                    enum("people.pose_activity.occlusions", person["occlusions"]),
                )
            if person.get("plane"):
                inc_vc(
                    output["people"]["positions"]["plane"],
                    enum("people.positions.plane", person["plane"]),
                )
            if person.get("side"):
                inc_vc(
                    output["people"]["positions"]["side"],
                    enum("people.positions.side", person["side"]),
                )
        objects = scene.get("objects") or []
        output["totals"]["object_instances"] += len(objects)
        for obj in objects:
            type_key = cluster("objects.type", obj.get("type", "unknown") or "unknown")
            subtype_key = cluster("objects.subtype", obj.get("subtype", "unknown") or "unknown")
            plane_key = enum("objects.plane", obj.get("plane", "unknown") or "unknown")
            side_key = enum("objects.side", obj.get("side", "unknown") or "unknown")

            inc_vc(output["objects"]["type"], type_key)
            inc_vc(output["objects"]["subtype"], subtype_key)
            inc_vc(output["objects"]["plane"], plane_key)
            inc_vc(output["objects"]["side"], side_key)

            type_bucket = ensure_type_bucket(output, type_key)
            type_bucket["count"] += 1
            type_bucket["by_subtype"][subtype_key] = int(
                type_bucket["by_subtype"].get(subtype_key, 0)
            ) + 1

            plane_side = f"{plane_key}.{side_key}"
            inc_vc(type_bucket["positions"]["plane_side"], plane_side)

            attributes = obj.get("attributes") or {}
            for key in ("colors", "material", "texture", "finish", "condition", "state", "size_class"):
                values = attributes.get(key)
                if values is None:
                    continue
                for value in _iter_clean(values):
                    if key == "colors":
                        inc_vc(type_bucket["attributes"]["colors"], normalize_color_token(value, config))
                    else:
                        inc_vc(
                            type_bucket["attributes"][key],
                            cluster(f"objects.attributes.{key}", value),
                        )

            quantity = obj.get("quantity") or {}
            exact = quantity.get("exact")
            if exact is not None:
                try:
                    exact_int = int(exact)
                    type_bucket["quantity"]["exact_sum"] += exact_int
                    type_bucket["quantity"]["exact_present_count"] += 1
                except Exception:
                    pass
            for value in _iter_clean(quantity.get("approx")):
                inc_vc(
                    type_bucket["quantity"]["approx_tokens"],
                    cluster("objects.quantity.approx_tokens", value),
                )

            pair_bucket = ensure_type_subtype_bucket(output, type_key, subtype_key)
            pair_bucket["count"] += 1
            inc_vc(pair_bucket["positions"]["plane_side"], plane_side)
            for key in ("colors", "material", "texture", "finish", "condition", "state", "size_class"):
                values = attributes.get(key)
                if values is None:
                    continue
                for value in _iter_clean(values):
                    if key == "colors":
                        inc_vc(pair_bucket["attributes"]["colors"], normalize_color_token(value, config))
                    else:
                        inc_vc(
                            pair_bucket["attributes"][key],
                            cluster(f"objects.attributes.{key}", value),
                        )
            if exact is not None:
                try:
                    exact_int = int(exact)
                    pair_bucket["quantity"]["exact_sum"] += exact_int
                    pair_bucket["quantity"]["exact_present_count"] += 1
                except Exception:
                    pass
            for value in _iter_clean(quantity.get("approx")):
                inc_vc(
                    pair_bucket["quantity"]["approx_tokens"],
                    cluster("objects.quantity.approx_tokens", value),
                )

            by_position = output["objects"]["by_position"]
            if plane_side not in by_position:
                by_position[plane_side] = {"type_counts": {}, "type_subtype_counts": {}}
            by_position[plane_side]["type_counts"][type_key] = int(
                by_position[plane_side]["type_counts"].get(type_key, 0)
            ) + 1
            pair_key = f"{type_key}|||{subtype_key}"
            by_position[plane_side]["type_subtype_counts"][pair_key] = int(
                by_position[plane_side]["type_subtype_counts"].get(pair_key, 0)
            ) + 1

            background_counts = output["background"]["object_counts_by_position"]
            background_counts.setdefault(plane_side, {"total": 0})
            background_counts[plane_side]["total"] += 1
        texts = scene.get("texts") or []
        output["totals"]["text_instances"] += len(texts)
        for text_item in texts:
            if text_item.get("content"):
                inc_vc(output["texts"]["content"], cluster("texts.content", text_item["content"]))
            for value in _iter_clean(text_item.get("font_style")):
                inc_vc(output["texts"]["font_style"], cluster("texts.font_style", value))
            for key in ("plane", "side", "legibility"):
                if text_item.get(key):
                    inc_vc(output["texts"][key], cluster(f"texts.{key}", text_item[key]))

        camera = scene.get("camera") or {}
        if camera.get("angle"):
            inc_vc(output["camera"]["angle"], enum("camera.angle", camera["angle"]))
        if camera.get("perspective"):
            inc_vc(output["camera"]["perspective"], cluster("camera.perspective", camera["perspective"]))
        if camera.get("focal_length"):
            inc_vc(output["camera"]["focal_length"], enum("camera.focal_length", camera["focal_length"]))
        if camera.get("depth_of_field"):
            inc_vc(output["camera"]["depth_of_field"], enum("camera.depth_of_field", camera["depth_of_field"]))
        for value in _iter_clean(camera.get("framing")):
            inc_vc(output["camera"]["framing"], cluster("camera.framing", value))
        for value in _iter_clean(camera.get("crop")):
            inc_vc(output["camera"]["crop"], cluster("camera.crop", value))

        lighting = scene.get("lighting") or {}
        for key in ("color_temperature", "contrast_level", "saturation_level"):
            if lighting.get(key):
                inc_vc(output["lighting"][key], enum(f"lighting.{key}", lighting[key]))
        for source in lighting.get("sources") or []:
            if source.get("type"):
                inc_vc(output["lighting"]["sources"]["type"], cluster("lighting.sources.type", source["type"]))
            if source.get("count"):
                inc_vc(output["lighting"]["sources"]["count"], cluster("lighting.sources.count", source["count"]))
            for value in _iter_clean(source.get("directionality")):
                inc_vc(
                    output["lighting"]["sources"]["directionality"],
                    cluster("lighting.sources.directionality", value),
                )
            if source.get("hardness"):
                inc_vc(
                    output["lighting"]["sources"]["hardness"],
                    cluster("lighting.sources.hardness", source["hardness"]),
                )
        for value in _iter_clean(lighting.get("shadows")):
            inc_vc(output["lighting"]["shadows"], cluster("lighting.shadows", value))
        for value in _iter_clean(lighting.get("artifacts")):
            inc_vc(output["lighting"]["artifacts"], cluster("lighting.artifacts", value))

        safety = scene.get("safety") or {}
        for value in _iter_clean(safety.get("hazards")):
            inc_vc(output["safety"]["hazards"], cluster("safety.hazards", value))
        for value in _iter_clean(safety.get("nsfw")):
            inc_vc(output["safety"]["nsfw"], cluster("safety.nsfw", value))
        for value in _iter_clean(scene.get("uncertainty")):
            inc_vc(output["uncertainty"], cluster("uncertainty", value))

    return output
def write_counts(counts: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(counts, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def run_counts(input_dir: Path, output_path: Path, cfg: Optional[FrequencyCounterConfig] = None) -> Path:
    counts = generate_counts(input_dir, cfg)
    return write_counts(counts, output_path)


__all__ = [
    "FrequencyCounterConfig",
    "generate_counts",
    "run_counts",
    "write_counts",
]
