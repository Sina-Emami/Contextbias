import base64
import mimetypes
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


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
ColorTemperature = Literal["warm", "neutral", "cool", "unknown"]
DominantPalette = Literal[
    "monochrome",
    "analogous",
    "complementary",
    "triadic",
    "tetradic",
    "earth_tones",
    "neon",
    "pastel",
    "high_key",
    "low_key",
    "unknown",
]
AtmosphereContrastLevel = Literal["low", "medium", "high", "unknown"]
AestheticQualities = Literal[
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
    "unknown",
]
SourcesCount = Literal["one", "two", "three", "four_or_more", "unknown"]
SourcesType = Literal["natural", "artificial", "mixed", "unknown"]
ContrastLevel = Literal["low", "medium", "high", "unknown"]
Directionality = Literal[
    "omnidirectional",
    "directional",
    "backlight",
    "sidelight",
    "toplight",
    "underlight",
    "unknown",
]
SaturationLevel = Literal["desaturated", "normal", "vibrant", "oversaturated", "unknown"]
Shadows = Literal["none", "soft", "medium", "hard", "casting", "unknown"]
Hardness = Literal["soft", "medium", "hard", "unknown"]
Perspective = Literal[
    "first_person",
    "third_person",
    "birdseye",
    "wormseye",
    "eye_level",
    "high_angle",
    "low_angle",
    "unknown",
]
Framing = Literal[
    "close_up",
    "medium",
    "full_body",
    "extreme_close_up",
    "headshot",
    "wide",
    "establishing",
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
TimeOfDayHint = Literal[
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
Hairstyle = Literal[
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
    "ombre",
    "highlight",
    "unknown",
]
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
FaceEmotion = Literal[
    "neutral",
    "happy",
    "sad",
    "angry",
    "surprised",
    "disgusted",
    "fearful",
    "confused",
    "smirking",
    "blushing",
    "frowning",
    "unknown",
]
ClothesColor = Literal[
    "white",
    "black",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "grey",
    "tan",
    "multi",
    "patterned",
    "unknown",
]
IndoorOutdoor = Literal["indoor", "outdoor", "unknown"]
GenderPresentation = Literal["female", "male", "nonbinary", "unknown"]
SkinTone = Literal["light", "medium", "dark", "unknown"]
AgeRange = Literal[
    "child",
    "teen",
    "young_adult",
    "middle_aged",
    "older_adult",
    "unknown",
]
PresenceTernary = Literal["yes", "no", "unknown"]
OrientationType = Literal["front", "three_quarter", "profile", "back", "unknown"]
PlaneType = Literal["foreground", "midground", "background", "unknown"]
SideType = Literal["left", "center", "right", "unknown"]


class AtmosphereInfo(BaseModel):
    mood: List[Mood] = Field(default_factory=list)
    dominant_palette: List[DominantPalette] = Field(default_factory=list)
    color_temperature: ColorTemperature = "unknown"
    contrast_level: AtmosphereContrastLevel = "unknown"
    saturation_level: SaturationLevel = "unknown"
    aesthetic_qualities: List[AestheticQualities] = Field(default_factory=list)


class SurfaceDetail(BaseModel):
    material: List[str] = Field(default_factory=list)
    texture: List[str] = Field(default_factory=list)
    color: List[str] = Field(default_factory=list)
    finish: List[str] = Field(default_factory=list)
    condition: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class SurfaceBundle(BaseModel):
    walls: List[SurfaceDetail] = Field(default_factory=list)
    floor: List[SurfaceDetail] = Field(default_factory=list)
    ceiling: List[SurfaceDetail] = Field(default_factory=list)


class SpatialLayout(BaseModel):
    depth: Optional[str] = None
    openness: Optional[str] = None
    aisle_width: Optional[str] = None


class EnvironmentInfo(BaseModel):
    location_type: Optional[str] = None
    indoor_outdoor: IndoorOutdoor = "unknown"
    time_of_day_hint: TimeOfDayHint = "unknown"
    weather: Weather = "unknown"
    surfaces: SurfaceBundle = SurfaceBundle()
    spatial_layout: SpatialLayout = SpatialLayout()


class GroomingInfo(BaseModel):
    present: PresenceTernary = "unknown"
    style: List[str] = Field(default_factory=list)
    color: List[str] = Field(default_factory=list)


class HairInfo(BaseModel):
    present: PresenceTernary = "unknown"
    style: List[Hairstyle] = Field(default_factory=list)
    color: List[HairColor] = Field(default_factory=list)


class EyewearInfo(BaseModel):
    present: PresenceTernary = "unknown"
    type: List[str] = Field(default_factory=list)
    frame_color: List[str] = Field(default_factory=list)


class HeadCoveringInfo(BaseModel):
    present: PresenceTernary = "unknown"
    type: List[str] = Field(default_factory=list)
    color: List[str] = Field(default_factory=list)


class GarmentInfo(BaseModel):
    garment_type: Optional[str] = None
    color: List[ClothesColor] = Field(default_factory=list)
    material: List[str] = Field(default_factory=list)
    texture: List[str] = Field(default_factory=list)
    fit_style: List[str] = Field(default_factory=list)
    pattern: List[str] = Field(default_factory=list)
    condition: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class PersonInfo(BaseModel):
    id: Optional[str] = None
    gender_presentation: GenderPresentation = "unknown"
    skin_tone: SkinTone = "unknown"
    age_range: AgeRange = "unknown"
    role_hint: Optional[str] = None
    hair: HairInfo = HairInfo()
    facial_hair: GroomingInfo = GroomingInfo()
    eyewear: EyewearInfo = EyewearInfo()
    head_covering: HeadCoveringInfo = HeadCoveringInfo()
    clothing: List[GarmentInfo] = Field(default_factory=list)
    pose: List[str] = Field(default_factory=list)
    activities: List[str] = Field(default_factory=list)
    gaze_direction: GazeDirection = "unknown"
    orientation: OrientationType = "unknown"
    face_emotion: FaceEmotion = "unknown"
    notes: Optional[str] = None


class QuantityInfo(BaseModel):
    exact: Optional[int] = None
    approx: Optional[str] = None


class ObjectAttributes(BaseModel):
    colors: List[str] = Field(default_factory=list)
    material: List[str] = Field(default_factory=list)
    texture: List[str] = Field(default_factory=list)
    finish: List[str] = Field(default_factory=list)
    condition: List[str] = Field(default_factory=list)
    state: List[str] = Field(default_factory=list)
    size_class: Optional[str] = None


class ObjectInfo(BaseModel):
    id: Optional[str] = None
    type: Optional[str] = None
    subtype: Optional[str] = None
    plane: PlaneType = "unknown"
    side: SideType = "unknown"
    attributes: ObjectAttributes = ObjectAttributes()
    quantity: QuantityInfo = QuantityInfo()
    notes: Optional[str] = None


class SideBuckets(BaseModel):
    left: List[str] = Field(default_factory=list)
    center: List[str] = Field(default_factory=list)
    right: List[str] = Field(default_factory=list)


class BackgroundLayout(BaseModel):
    foreground: SideBuckets = SideBuckets()
    midground: SideBuckets = SideBuckets()
    background: SideBuckets = SideBuckets()


class SceneText(BaseModel):
    content: Optional[str] = None
    font_style: List[str] = Field(default_factory=list)
    plane: PlaneType = "unknown"
    side: SideType = "unknown"
    legibility: Optional[str] = None
    notes: Optional[str] = None


class CameraInfo(BaseModel):
    angle: CameraAngle = "unknown"
    perspective: Perspective = "unknown"
    focal_length: FocalLength = "unknown"
    depth_of_field: DepthOfField = "unknown"
    framing: List[Framing] = Field(default_factory=list)
    crop: List[str] = Field(default_factory=list)


class LightingSourceInfo(BaseModel):
    type: SourcesType = "unknown"
    count: SourcesCount = "unknown"
    directionality: List[Directionality] = Field(default_factory=list)
    hardness: Hardness = "unknown"


class LightingInfo(BaseModel):
    sources: List[LightingSourceInfo] = Field(default_factory=list)
    shadows: List[Shadows] = Field(default_factory=list)
    artifacts: List[str] = Field(default_factory=list)


class SafetyInfo(BaseModel):
    hazards: List[str] = Field(default_factory=list)
    nsfw: List[str] = Field(default_factory=list)


class SceneInfo(BaseModel):
    summary: Optional[str] = None
    atmosphere: AtmosphereInfo = AtmosphereInfo()
    environment: EnvironmentInfo = EnvironmentInfo()
    people: List[PersonInfo] = Field(default_factory=list)
    objects: List[ObjectInfo] = Field(default_factory=list)
    background: BackgroundLayout = BackgroundLayout()
    texts: List[SceneText] = Field(default_factory=list)
    camera: CameraInfo = CameraInfo()
    lighting: LightingInfo = LightingInfo()
    safety: SafetyInfo = SafetyInfo()
    uncertainty: List[str] = Field(default_factory=list)


class ImageAuditRecord(BaseModel):
    image_id: Optional[str] = None
    image_path: Optional[str] = None
    scene: SceneInfo = SceneInfo()


def file_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "application/octet-stream"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"