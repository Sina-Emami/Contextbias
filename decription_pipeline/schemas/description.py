"""Structured schema for image descriptions used across the description pipeline."""

import base64
import mimetypes
from typing import Dict, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

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
AtmosphereContrastLevel = Literal["low", "medium", "high", "unknown"]
SaturationLevel = Literal["desaturated", "neutral", "vibrant", "oversaturated", "unknown"]
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
    "grey",
    "silver",
    "gold",
    "beige",
    "brown",
    "tan",
    "cream",
    "ivory",
    "red",
    "orange",
    "yellow",
    "green",
    "teal",
    "turquoise",
    "blue",
    "navy",
    "purple",
    "pink",
    "magenta",
    "maroon",
    "burgundy",
    "olive",
    "cyan",
    "multi",
    "unknown",
]
DominantPalette = Literal[
    "monochrome",
    "analogous",
    "complementary",
    "split_complementary",
    "triadic",
    "tetradic",
    "earth_tones",
    "neon",
    "pastel",
    "high_key",
    "low_key",
    "duotone",
    "unknown",
]

ScenePlane = Literal["foreground", "midground", "background", "unknown"]
SceneSide = Literal["left", "center", "right", "unknown"]
ScenePosition = Literal[
    "foreground_left",
    "foreground_center",
    "foreground_right",
    "midground_left",
    "midground_center",
    "midground_right",
    "background_left",
    "background_center",
    "background_right",
    "unknown",
]

CameraAngle = Literal[
    "straight",
    "tilt_left",
    "tilt_right",
    "pan_left",
    "pan_right",
    "dolly_in",
    "dolly_out",
    "zoom_in",
    "zoom_out",
    "overhead",
    "low",
    "high",
    "unknown",
]
Perspective = Literal[
    "first_person",
    "third_person",
    "birdseye",
    "wormseye",
    "eye_level",
    "high_angle",
    "low_angle",
    "isometric",
    "unknown",
]
FocalLength = Literal[
    "ultra_wide",
    "wide",
    "standard",
    "telephoto",
    "supertelephoto",
    "macro",
    "unknown",
]
DepthOfField = Literal["shallow", "medium", "deep", "infinite", "unknown"]
Framing = Literal[
    "extreme_close_up",
    "close_up",
    "medium",
    "full_body",
    "wide",
    "establishing",
    "portrait",
    "landscape",
    "unknown",
]
CropType = Literal[
    "tight",
    "loose",
    "square",
    "portrait",
    "landscape",
    "letterbox",
    "pillarbox",
    "unknown",
]

IndoorOutdoor = Literal["indoor", "outdoor", "unknown"]
LocationType = Literal[
    "residential",
    "office",
    "hospital",
    "education",
    "industrial",
    "retail",
    "public_space",
    "transport",
    "landscape",
    "worship",
    "sports",
    "laboratory",
    "studio",
    "kitchen",
    "bathroom",
    "bedroom",
    "living_room",
    "unknown",
]
Openness = Literal["enclosed", "semi_open", "open", "expansive", "unknown"]
SpatialDepth = Literal["shallow", "moderate", "deep", "unknown"]
SpatialAisleWidth = Literal["none", "narrow", "standard", "wide", "extra_wide", "unknown"]
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
TimeOfDay = Literal[
    "dawn",
    "morning",
    "noon",
    "afternoon",
    "dusk",
    "evening",
    "night",
    "midnight",
    "unknown",
]

MaterialType = Literal[
    "wood",
    "metal",
    "glass",
    "fabric",
    "plastic",
    "stone",
    "brick",
    "concrete",
    "tile",
    "carpet",
    "leather",
    "paper",
    "ceramic",
    "organic",
    "vegetation",
    "water",
    "composite",
    "unknown",
]
TextureType = Literal[
    "smooth",
    "rough",
    "matte",
    "glossy",
    "patterned",
    "woven",
    "gritty",
    "polished",
    "weathered",
    "metallic",
    "reflective",
    "translucent",
    "textured",
    "coarse",
    "soft",
    "unknown",
]

Directionality = Literal[
    "omnidirectional",
    "directional",
    "backlight",
    "sidelight",
    "toplight",
    "underlight",
    "unknown",
]
Hardness = Literal["soft", "medium", "hard", "unknown"]
Shadows = Literal["none", "soft", "medium", "hard", "casting", "unknown"]
SourcesCount = Literal["one", "two", "three", "four_or_more", "unknown"]
SourcesType = Literal["natural", "artificial", "mixed", "unknown"]
LightingArtifact = Literal[
    "glare",
    "lens_flare",
    "noise",
    "reflection",
    "shadow_band",
    "chromatic_aberration",
    "light_streak",
    "specular_highlight",
    "none",
    "unknown",
]

ObjectPlacement = Literal["foreground", "midground", "background", "unknown"]
ObjectPosition = Literal["left", "center", "right", "overhead", "unknown"]
ObjectSize = Literal["tiny", "small", "medium", "large", "massive", "unknown"]
ObjectState = Literal[
    "new",
    "worn",
    "damaged",
    "broken",
    "clean",
    "dirty",
    "open",
    "closed",
    "powered_on",
    "powered_off",
    "active",
    "inactive",
    "unknown",
]
ObjectType = Literal[
    "furniture",
    "appliance",
    "device",
    "tool",
    "decoration",
    "lighting",
    "food",
    "beverage",
    "container",
    "vehicle",
    "textile",
    "structure",
    "artwork",
    "signage",
    "instrument",
    "weapon",
    "document",
    "equipment",
    "natural",
    "unknown",
]

AccessoryType = Literal[
    "bag",
    "belt",
    "bracelet",
    "earrings",
    "gloves",
    "hat",
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
    "eyewear",
    "none",
    "unknown",
]
ActivityType = Literal[
    "standing",
    "sitting",
    "walking",
    "running",
    "talking",
    "working",
    "reading",
    "writing",
    "using_device",
    "holding_object",
    "gesturing",
    "posing",
    "resting",
    "observing",
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
    "off_center",
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
    "auburn",
    "chestnut",
    "blue",
    "green",
    "pink",
    "purple",
    "silver",
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
    "cap",
    "helmet",
    "hood",
    "hijab",
    "turban",
    "veil",
    "scarf",
    "headband",
    "beanie",
    "hairnet",
    "none",
    "unknown",
]
ClothingGarment = Literal[
    "shirt",
    "t_shirt",
    "blouse",
    "dress",
    "jacket",
    "coat",
    "sweater",
    "hoodie",
    "pants",
    "jeans",
    "shorts",
    "skirt",
    "suit",
    "tie",
    "uniform",
    "scrubs",
    "lab_coat",
    "apron",
    "vest",
    "hat",
    "helmet",
    "gloves",
    "shoes",
    "boots",
    "sneakers",
    "scarf",
    "belt",
    "unknown",
]
ClothingMaterial = Literal[
    "cotton",
    "denim",
    "linen",
    "wool",
    "fleece",
    "silk",
    "velvet",
    "leather",
    "suede",
    "knit",
    "lace",
    "mesh",
    "corduroy",
    "synthetic",
    "unknown",
]
ClothingTexture = Literal[
    "smooth",
    "matte",
    "glossy",
    "ribbed",
    "knit",
    "quilted",
    "wrinkled",
    "fuzzy",
    "mesh",
    "lace",
    "sheer",
    "denim_twill",
    "corduroy_wale",
    "stripe",
    "unknown",
]
PoseType = Literal[
    "standing",
    "sitting",
    "walking",
    "running",
    "kneeling",
    "lying",
    "leaning",
    "reaching",
    "bending",
    "arms_crossed",
    "hands_on_hips",
    "unknown",
]
RoleHint = Literal[
    "doctor",
    "nurse",
    "patient",
    "teacher",
    "student",
    "athlete",
    "worker",
    "manager",
    "performer",
    "customer",
    "driver",
    "passenger",
    "parent",
    "child",
    "law_enforcement",
    "military",
    "scientist",
    "unknown",
]

SafetyHazard = Literal[
    "none",
    "slip",
    "trip",
    "fire",
    "electrical",
    "chemical",
    "biohazard",
    "sharp_object",
    "weapon",
    "crowd",
    "unknown",
]
SafetyNSFW = Literal[
    "none",
    "suggestive",
    "explicit",
    "gore",
    "violence",
    "self_harm",
    "unknown",
]

FontStyle = Literal[
    "serif",
    "sans_serif",
    "slab_serif",
    "script",
    "decorative",
    "handwritten",
    "monospace",
    "block",
    "unknown",
]
Legibility = Literal["clear", "partial", "poor", "illegible", "unknown"]

IntOrUnknown = Union[int, Literal["unknown"]]


class StrictBaseModel(BaseModel):
    """Apply consistent validation across all schema models."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"additionalProperties": False})


# ---------------------------------------------------------------------------
# Cohort payloads
# ---------------------------------------------------------------------------


class AtmosphereCohort(StrictBaseModel):
    aesthetic_qualities: List[AestheticQuality] = Field(default_factory=list)
    color_temperature: ColorTemperature = "unknown"
    contrast_level: AtmosphereContrastLevel = "unknown"
    dominant_color: List[ColorName] = Field(default_factory=list)
    dominant_palette: List[DominantPalette] = Field(default_factory=list)
    mood: Mood = "unknown"
    saturation_level: SaturationLevel = "unknown"


class BackgroundPositionCount(StrictBaseModel):
    plane: ScenePlane = "unknown"
    side: SceneSide = "unknown"
    count: IntOrUnknown = Field(default="unknown")


class BackgroundTotalsByPosition(StrictBaseModel):
    position: ScenePosition = "unknown"
    count: IntOrUnknown = Field(default="unknown")


class BackgroundCohort(StrictBaseModel):
    object_counts_by_position: List[BackgroundPositionCount] = Field(default_factory=list)
    position: ScenePosition = "unknown"
    totals_by_position: List[BackgroundTotalsByPosition] = Field(default_factory=list)


class CameraCohort(StrictBaseModel):
    angle: CameraAngle = "unknown"
    crop: List[CropType] = Field(default_factory=list)
    depth_of_field: DepthOfField = "unknown"
    focal_length: FocalLength = "unknown"
    framing: Framing = "unknown"
    perspective: Perspective = "unknown"


class EnvironmentCohort(StrictBaseModel):
    ceiling_color: ColorName = "unknown"
    ceiling_material: MaterialType = "unknown"
    ceiling_texture: TextureType = "unknown"
    floor_color: ColorName = "unknown"
    floor_material: MaterialType = "unknown"
    floor_texture: TextureType = "unknown"
    indoor_outdoor: IndoorOutdoor = "unknown"
    location_type: LocationType = "unknown"
    openness: Openness = "unknown"
    spatial_depth: SpatialDepth = "unknown"
    spatial_layout_aisle_width: SpatialAisleWidth = "unknown"
    spatial_layout_depth: SpatialDepth = "unknown"
    spatial_layout_openness: Openness = "unknown"
    time_of_day: TimeOfDay = "unknown"
    wall_color: ColorName = "unknown"
    wall_material: List[MaterialType] = Field(default_factory=list)
    wall_texture: TextureType = "unknown"
    weather: Weather = "unknown"


class LightingCohort(StrictBaseModel):
    artifacts: List[LightingArtifact] = Field(default_factory=list)
    directionality: Directionality = "unknown"
    hardness: Hardness = "unknown"
    shadows: Shadows = "unknown"
    sources_count: SourcesCount = "unknown"
    sources_type: SourcesType = "unknown"


class ObjectsCohort(StrictBaseModel):
    color: List[ColorName] = Field(default_factory=list)
    material: List[MaterialType] = Field(default_factory=list)
    placement: ObjectPlacement = "unknown"
    position: ObjectPosition = "unknown"
    size: ObjectSize = "unknown"
    state: List[ObjectState] = Field(default_factory=list)
    texture: List[TextureType] = Field(default_factory=list)
    type: List[ObjectType] = Field(default_factory=list)


class PeopleCohort(StrictBaseModel):
    accessories: List[AccessoryType] = Field(default_factory=list)
    accessories_color: List[ColorName] = Field(default_factory=list)
    activities: List[ActivityType] = Field(default_factory=list)
    age_range: AgeRange = "unknown"
    body_type: BodyType = "unknown"
    clothing_color: List[ColorName] = Field(default_factory=list)
    clothing_garment: List[ClothingGarment] = Field(default_factory=list)
    clothing_material: List[ClothingMaterial] = Field(default_factory=list)
    clothing_texture: List[ClothingTexture] = Field(default_factory=list)
    eyewear_present: PresenceValue = "unknown"
    eyewear_type: List[EyewearType] = Field(default_factory=list)
    facial_hair_present: PresenceValue = "unknown"
    facial_hair_style: List[FacialHairStyle] = Field(default_factory=list)
    facial_hair_color: List[ColorName] = Field(default_factory=list)
    gender_presentation: GenderPresentation = "unknown"
    gaze_direction: GazeDirection = "unknown"
    hair_present: PresenceValue = "unknown"
    hair_color: HairColor = "unknown"
    hair_style: HairStyle = "unknown"
    head_covering_present: PresenceValue = "unknown"
    head_covering_type: List[HeadCoveringType] = Field(default_factory=list)
    head_covering_color: List[ColorName] = Field(default_factory=list)
    pose: PoseType = "unknown"
    role_hint: RoleHint = "unknown"
    skin_tone: SkinTone = "unknown"


class SafetyCohort(StrictBaseModel):
    hazards: SafetyHazard = "unknown"
    nsfw: SafetyNSFW = "unknown"


class TextsCohort(StrictBaseModel):
    font_style: FontStyle = "unknown"
    legibility: Legibility = "unknown"
    plane: List[ScenePlane] = Field(default_factory=list)
    side: SceneSide = "unknown"


class TotalsCohort(StrictBaseModel):
    images: IntOrUnknown = Field(default="unknown")
    object_instances: IntOrUnknown = Field(default="unknown")
    people_instances: IntOrUnknown = Field(default="unknown")
    text_instances: IntOrUnknown = Field(default="unknown")


class UncertaintyCohort(StrictBaseModel):
    note: List[str] = Field(default_factory=list)


class CohortBundle(StrictBaseModel):
    atmosphere: AtmosphereCohort = Field(default_factory=AtmosphereCohort)
    background: BackgroundCohort = Field(default_factory=BackgroundCohort)
    camera: CameraCohort = Field(default_factory=CameraCohort)
    environment: EnvironmentCohort = Field(default_factory=EnvironmentCohort)
    lighting: LightingCohort = Field(default_factory=LightingCohort)
    objects: ObjectsCohort = Field(default_factory=ObjectsCohort)
    people: PeopleCohort = Field(default_factory=PeopleCohort)
    safety: SafetyCohort = Field(default_factory=SafetyCohort)
    texts: TextsCohort = Field(default_factory=TextsCohort)
    totals: TotalsCohort = Field(default_factory=TotalsCohort)
    uncertainty: UncertaintyCohort = Field(default_factory=UncertaintyCohort)


# ---------------------------------------------------------------------------
# Top-level schema
# ---------------------------------------------------------------------------


class ImageInfo(StrictBaseModel):
    image_id: str = Field(..., min_length=1)
    source_url: str = ""
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
    "AtmosphereCohort",
    "BackgroundCohort",
    "CameraCohort",
    "EnvironmentCohort",
    "LightingCohort",
    "ObjectsCohort",
    "PeopleCohort",
    "SafetyCohort",
    "TextsCohort",
    "TotalsCohort",
    "UncertaintyCohort",
]
