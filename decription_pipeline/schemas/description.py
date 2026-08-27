"""Structured schema for image descriptions used across the description pipeline."""

import base64
import mimetypes
from typing import Dict, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


Mood = Literal[
    "calm",
    "tense",
    "joyful",
    "melancholic",
    "mysterious",
    "romantic",
    "foreboding",
    "whimsical",
    "minimalistic",
    "dramatic",
    "unknown",
]

ColorTemperature = Literal["warm", "neutral", "cool", "mixed", "unknown"]
AestheticQuality = Literal[
    "minimal",
    "baroque",
    "retro",
    "surreal",
    "abstract",
    "realistic",
    "fantasy",
    "cinematic",
    "gritty",
    "dreamy",
    "industrial",
    "unknown",
]

ColorName = Literal[
    "white",
    "black",
    "gray",
    "silver",
    "gold",
    "beige",
    "brown",
    "tan",
    "cream",
    "red",
    "orange",
    "yellow",
    "green",
    "teal",
    "blue",
    "navy",
    "purple",
    "pink",
    "olive",
    "unknown",
]

Perspective = Literal[
    "first_person",
    "third_person",
    "birdseye",
    "wormseye",
    "eye_level",
    "high_angle",
    "unknown",
]

DepthOfField = Literal["shallow", "medium", "deep", "infinite", "unknown"]
Framing = Literal["tight", "medium", "wide", "balanced", "asymmetric", "unknown"]

AccessoryType = Literal[
    "bag",
    "belt",
    "bracelet",
    "earrings",
    "gloves",
    "headphones",
    "jewelry",
    "mask",
    "necklace",
    "scarf",
    "watch",
    "badge",
    "tie",
    "tool",
    "safety_gear",
    "medical",
    "religious",
    "none",
    "unknown",
]

BodyType = Literal["slim", "average", "athletic", "stocky", "curvy", "unknown"]
GenderPresentation = Literal["female", "male", "nonbinary", "unknown"]
SkinTone = Literal["light", "medium", "dark", "unknown"]
AgeRange = Literal["child", "teen", "young_adult", "middle_aged", "older_adult", "unknown"]
PresenceValue = Literal["yes", "no", "unknown"]

GazeDirection = Literal[
    "forward",
    "left",
    "right",
    "up",
    "down",
    "away",
    "toward_camera",
    "unknown",
]

HairStyle = Literal[
    "short",
    "long",
    "bob",
    "pixie",
    "ponytail",
    "braids",
    "bun",
    "curly",
    "wavy",
    "straight",
    "updo",
    "shaved",
    "afro",
    "mohawk",
    "unknown",
]

HairColor = Literal[
    "black",
    "brown",
    "blonde",
    "red",
    "grey",
    "white",
    "blue",
    "green",
    "pink",
    "purple",
    "unknown",
]

EyewearType = Literal[
    "eyeglasses",
    "sunglasses",
    "safety_goggles",
    "ski_goggles",
    "swim_goggles",
    "monocle",
    "vr_headset",
    "protective_face_shield",
    "none",
    "unknown",
]

FacialHairStyle = Literal[
    "mustache",
    "beard",
    "goatee",
    "stubble",
    "soul_patch",
    "sideburns",
    "none",
    "unknown",
]

HeadCoveringType = Literal[
    "hat",
    "helmet",
    "hood",
    "hijab",
    "turban",
    "veil",
    "headband",
    "beanie",
    "hairnet",
    "none",
    "unknown",
]

Expression = Literal[
    "neutral",
    "smiling",
    "frowning",
    "serious",
    "focused",
    "surprised",
    "angry",
    "sad",
    "happy",
    "tired",
    "unknown",
]

PoseType = Literal[
    "standing",
    "sitting",
    "kneeling",
    "lying",
    "bending",
    "arms_crossed",
    "hands_on_hips",
    "unknown",
]

Weather = Literal[
    "clear",
    "sunny",
    "cloudy",
    "overcast",
    "rain",
    "drizzle",
    "storm",
    "snow",
    "fog",
    "mist",
    "windy",
    "hail",
    "unknown",
]

IntOrUnknown = Union[int, Literal["unknown"]]


class StrictBaseModel(BaseModel):
    """Apply consistent validation across all schema models."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"additionalProperties": False})


class SceneAppearanceCohort(StrictBaseModel):
    """Overall look and feel of the scene, including ambient colors."""
    mood: Mood = "unknown"
    color_temperature: ColorTemperature = "unknown"
    aesthetic_qualities: List[AestheticQuality] = Field(default_factory=list)
    dominant_colors: List[ColorName] = Field(default_factory=list)
    weather: Weather = "unknown"


class CameraCohort(StrictBaseModel):
    depth_of_field: DepthOfField = "unknown"
    framing: Framing = "unknown"
    perspective: Perspective = "unknown"


class ObjectsCohort(StrictBaseModel):
    items: List[str] = Field(default_factory=list, max_length=3)


class PersonClothing(StrictBaseModel):
    clothing_garment: str = "unknown"
    clothing_color: List[ColorName] = Field(default_factory=list)


class PersonAttributes(StrictBaseModel):
    person_id: str = ""
    accessories: List[AccessoryType] = Field(default_factory=list)
    activities: List[str] = Field(default_factory=list)
    age_range: AgeRange = "unknown"
    body_type: BodyType = "unknown"
    clothing: List[PersonClothing] = Field(default_factory=list)
    eyewear_present: PresenceValue = "unknown"
    eyewear_type: EyewearType = "unknown"
    facial_hair_present: PresenceValue = "unknown"
    facial_hair_style: FacialHairStyle = "unknown"
    facial_hair_color: ColorName = "unknown"
    gender_presentation: GenderPresentation = "unknown"
    gaze_direction: GazeDirection = "unknown"
    hair_present: PresenceValue = "unknown"
    hair_color: HairColor = "unknown"
    hair_style: HairStyle = "unknown"
    tattoo_present: PresenceValue = "unknown"
    head_covering_present: PresenceValue = "unknown"
    head_covering_type: HeadCoveringType = "unknown"
    pose: PoseType = "unknown"
    expression: Expression = "unknown"
    skin_tone: SkinTone = "unknown"


class PeopleCohort(StrictBaseModel):
    persons: List[PersonAttributes] = Field(default_factory=list)


class TotalsCohort(StrictBaseModel):
    images: IntOrUnknown = Field(default="unknown")
    object_instances: IntOrUnknown = Field(default="unknown")
    people_instances: IntOrUnknown = Field(default="unknown")


class CohortBundle(StrictBaseModel):
    scene_appearance: SceneAppearanceCohort = Field(default_factory=SceneAppearanceCohort)
    camera: CameraCohort = Field(default_factory=CameraCohort)
    objects: ObjectsCohort = Field(default_factory=ObjectsCohort)
    people: PeopleCohort = Field(default_factory=PeopleCohort)
    totals: TotalsCohort = Field(default_factory=TotalsCohort)


class ImageInfo(StrictBaseModel):
    image_id: str = Field(..., min_length=1)
    file_path: str = ""
    source_model: str = ""
    caption: str = ""
    source_metadata: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict)


class ImageAuditRecord(StrictBaseModel):
    schema_version: Literal["1.0"] = Field(default=SCHEMA_VERSION)
    image: ImageInfo = Field(default_factory=lambda: ImageInfo(image_id="unknown"))
    cohorts: CohortBundle = Field(default_factory=CohortBundle)


def file_to_data_url(path: str) -> str:
    """Read a path and convert it to a data URL for LLM vision inputs."""

    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "application/octet-stream"
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


__all__ = [
    "SCHEMA_VERSION",
    "ImageAuditRecord",
    "ImageInfo",
    "file_to_data_url",
    "CohortBundle",
    "SceneAppearanceCohort",
    "CameraCohort",
    "ObjectsCohort",
    "PeopleCohort",
    "TotalsCohort",
]
