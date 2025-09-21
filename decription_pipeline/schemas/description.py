import base64
import mimetypes
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


ColorTemperature = Literal["warm", "neutral", "cool", "unknown"]
ContrastLevel = Literal["low", "medium", "high", "unknown"]
SaturationLevel = Literal["muted", "neutral", "vivid", "unknown"]
IndoorOutdoor = Literal["indoor", "outdoor", "unknown"]
GenderPresentation = Literal["female", "male", "nonbinary", "unknown"]
SkinTone = Literal["light", "medium", "dark", "unknown"]
AgeRange = Literal["child", "teen", "young_adult", "middle_aged", "older_adult", "unknown"]
PresenceTernary = Literal["yes", "no", "unknown"]
GazeDirection = Literal["left", "center", "right", "up", "down", "unknown"]
OrientationType = Literal["front", "three_quarter", "profile", "back", "unknown"]
OcclusionState = Literal["none", "partial", "significant", "unknown"]
PlaneType = Literal["foreground", "midground", "background", "unknown"]
SideType = Literal["left", "center", "right", "unknown"]
CameraAngle = Literal["high", "eye_level", "low", "unknown"]
DepthOfField = Literal["shallow", "moderate", "deep", "unknown"]
LightingHardness = Literal["soft", "medium", "hard", "unknown"]


class LightingProfile(BaseModel):
    color_temperature: ColorTemperature = "unknown"
    contrast_level: ContrastLevel = "unknown"
    saturation_level: SaturationLevel = "unknown"
    aesthetic_qualities: List[str] = Field(default_factory=list)


class AtmosphereInfo(BaseModel):
    mood: List[str] = Field(default_factory=list)
    dominant_palette: List[str] = Field(default_factory=list)
    lighting_profile: LightingProfile = LightingProfile()


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
    time_of_day_hint: Optional[str] = None
    weather: Optional[str] = None
    surfaces: SurfaceBundle = SurfaceBundle()
    spatial_layout: SpatialLayout = SpatialLayout()


class GroomingInfo(BaseModel):
    present: PresenceTernary = "unknown"
    style: List[str] = Field(default_factory=list)
    color: List[str] = Field(default_factory=list)


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
    color: List[str] = Field(default_factory=list)
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
    hair: GroomingInfo = GroomingInfo()
    facial_hair: GroomingInfo = GroomingInfo()
    eyewear: EyewearInfo = EyewearInfo()
    head_covering: HeadCoveringInfo = HeadCoveringInfo()
    clothing: List[GarmentInfo] = Field(default_factory=list)
    pose: List[str] = Field(default_factory=list)
    activities: List[str] = Field(default_factory=list)
    gaze_direction: GazeDirection = "unknown"
    orientation: OrientationType = "unknown"
    occlusions: OcclusionState = "unknown"
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
    perspective: Optional[str] = None
    focal_length: Optional[str] = None
    depth_of_field: DepthOfField = "unknown"
    framing: List[str] = Field(default_factory=list)
    crop: List[str] = Field(default_factory=list)


class LightingSourceInfo(BaseModel):
    type: Optional[str] = None
    count: Optional[str] = None
    directionality: List[str] = Field(default_factory=list)
    hardness: LightingHardness = "unknown"


class LightingInfo(BaseModel):
    color_temperature: ColorTemperature = "unknown"
    contrast_level: ContrastLevel = "unknown"
    saturation_level: SaturationLevel = "unknown"
    sources: List[LightingSourceInfo] = Field(default_factory=list)
    shadows: List[str] = Field(default_factory=list)
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