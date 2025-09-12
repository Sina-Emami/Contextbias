import base64
import mimetypes
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field, conint, confloat

# ===== Enums =====
AgeBucket = Literal["child","teen","young_adult","middle_aged","older_adult","unknown"]
GenderPresentation = Literal["female","male","nonbinary","ambiguous","unknown"]
SkinToneLabel = Literal["very_light","light","medium","tan","brown","dark","very_dark","unknown"]
Position2D = Literal["left","center","right","unknown"]
DepthPlane = Literal["foreground","midground","background","unknown"]
Presence = Literal["present","absent","unknown"]
BoolUnknown = Literal["yes","no","unknown"]

HeightBucket = Literal["very_short","short","average","tall","very_tall","unknown"]
BodyBuild = Literal["slender","average","heavyset","muscular","pregnant_presenting","unknown"]
BodyPosition = Literal["standing","sitting","kneeling","lying","crouching","running","walking","unknown"]
Orientation = Literal["facing_camera","profile_left","profile_right","back_to_camera","three_quarter_left","three_quarter_right","unknown"]
Handedness = Literal["left","right","both","unknown"]
ArtStyle = Literal["photorealistic","illustration","3d_render","cartoon_anime","oil_painting","sketch","other","unknown"]

RaceEthnicityLabel = Literal[
    "white","black","east_asian","southeast_asian","indian","middle_eastern","latino_hispanic",
    "other","ambiguous","unknown"
]
ReligionLabel = Literal[
    "christian","muslim","jewish","hindu","buddhist","sikh","secular_none","other","unknown"
]
HeadCovering = Literal["none","hijab","niqab","turban","kippah","headscarf","hat","helmet","other","unknown"]

ColorBasic = Literal[
    "black","white","gray","red","orange","yellow","green","blue","purple","pink",
    "brown","beige","tan","navy","maroon","gold","silver","teal","cyan","magenta",
    "cream","unknown"
]

_ALLOWED = {
    "black","white","gray","red","orange","yellow","green","blue","purple","pink",
    "brown","beige","tan","navy","maroon","gold","silver","teal","cyan","magenta","cream"
}

_SYNONYMS = {
    "burgundy":"maroon","deep burgundy":"maroon","wine":"maroon",
    "navy blue":"navy","dark blue":"navy",
    "sky blue":"blue","royal blue":"blue","midnight blue":"navy",
    "charcoal":"gray","light gray":"gray","dark gray":"gray","silver gray":"silver",
    "ivory":"cream","off-white":"cream",
    "rose":"pink","hot pink":"pink","fuchsia":"magenta",
    "aqua":"cyan","turquoise":"teal",
    "olive":"green","lime":"green","forest green":"green",
    "amber":"yellow","golden":"gold",
    "tan brown":"tan","khaki":"tan",
    "beige tan":"beige","sand":"beige",
    "crimson":"red","scarlet":"red","cherry red":"red",
    "lavender":"purple","violet":"purple",
    "bronze":"brown","copper":"brown",
}

def normalize_color(text: Optional[str]) -> ColorBasic:
    if not text:
        return "unknown"  # type: ignore[return-value]
    t = text.strip().lower()
    if t in _ALLOWED:
        return t  # type: ignore[return-value]
    if t in _SYNONYMS:
        return _SYNONYMS[t]  # type: ignore[return-value]
    for prefix in ("dark ","light ","deep ","soft ","muted ","bright "):
        if t.startswith(prefix):
            base = t[len(prefix):]
            if base in _ALLOWED:
                return base  # type: ignore[return-value]
            if base in _SYNONYMS:
                return _SYNONYMS[base]  # type: ignore[return-value]
    return "unknown"  # type: ignore[return-value]

# ===== Schema =====
class RoleEvidence(BaseModel):
    role: str
    confidence: confloat(ge=0, le=1)
    evidence: List[str] = Field(default_factory=list)

class SensitiveEvidence(BaseModel):
    label: str
    confidence: confloat(ge=0, le=1)
    evidence: List[str] = Field(default_factory=list)
    policy: str = Field("evidence_only")

class ScriptInfo(BaseModel):
    iso15924: Optional[str] = None
    name: Optional[str] = None
    confidence: confloat(ge=0, le=1) = 0.0
    evidence: Optional[str] = None

class Person(BaseModel):
    person_id: str
    age_bucket: AgeBucket
    gender_presentation: GenderPresentation
    skin_tone_label: SkinToneLabel = "unknown"
    mst_skin_tone: Optional[conint(ge=1, le=10)] = None
    fitzpatrick_type: Optional[conint(ge=1, le=6)] = None

    height_bucket: HeightBucket = "unknown"
    body_build: BodyBuild = "unknown"
    body_position: BodyPosition = "unknown"
    orientation: Orientation = "unknown"
    handedness: Handedness = "unknown"

    hair: Optional[str] = None
    eyewear: str = "unknown"
    facial_hair: Optional[str] = None
    head_covering: HeadCovering = "unknown"

    clothing_items: List[str] = Field(default_factory=list)
    accessories: List[str] = Field(default_factory=list)
    visible_tattoos: BoolUnknown = "unknown"
    jewelry: List[str] = Field(default_factory=list)

    race_ethnicity_label: RaceEthnicityLabel = "unknown"
    race_ethnicity_evidence: Optional[SensitiveEvidence] = None
    religion_label: ReligionLabel = "unknown"
    religion_evidence: Optional[SensitiveEvidence] = None
    disability_indicators: List[str] = Field(default_factory=list)
    disability_evidence: Optional[SensitiveEvidence] = None
    language_script: Optional[ScriptInfo] = None

    position_horizontal: Position2D = "unknown"
    position_depth: DepthPlane = "unknown"

    posture: Optional[str] = None
    expression: Optional[str] = None
    role_inferred: Optional[RoleEvidence] = None
    skin_tone_notes: Optional[str] = None

class ObjectItem(BaseModel):
    name_raw: str
    name_canonical: Optional[str] = None
    attributes: Dict[str, str] = Field(default_factory=dict)
    count: int = 1

    colors: List[ColorBasic] = Field(default_factory=list)
    materials: List[str] = Field(default_factory=list)
    logo_text: Optional[str] = None
    brand_name: Optional[str] = None
    text_scripts: List[ScriptInfo] = Field(default_factory=list)

    position_horizontal: Position2D = "unknown"
    position_depth: DepthPlane = "unknown"

class EnvironmentElement(BaseModel):
    label_raw: str
    label_canonical: Optional[str] = None
    attributes: Dict[str, str] = Field(default_factory=dict)
    colors: List[ColorBasic] = Field(default_factory=list)
    position_horizontal: Position2D = "unknown"
    position_depth: DepthPlane = "unknown"

class Atmosphere(BaseModel):
    dominant_colors: List[ColorBasic] = Field(default_factory=list)
    color_temperature: Optional[str] = None
    contrast_level: Optional[str] = None
    saturation_level: Optional[str] = None
    mood: Optional[str] = None
    aesthetics: List[str] = Field(default_factory=list)

class CompositionLighting(BaseModel):
    camera_angle: Optional[str] = None
    framing: Optional[str] = None
    focal_length_feel: Optional[str] = None
    depth_of_field: Optional[str] = None
    symmetry_or_leading_lines: Optional[str] = None
    light_sources: Optional[str] = None
    shadow_characteristics: Optional[str] = None
    art_style: ArtStyle = "unknown"

class Safety(BaseModel):
    safety_indicators: List[str] = Field(default_factory=list)
    nsfw_indicators: List[str] = Field(default_factory=list)
    watermark_present: BoolUnknown = "unknown"

class Uncertainty(BaseModel):
    statements: List[str] = Field(default_factory=list)

class Setting(BaseModel):
    indoor_outdoor: str = "unknown"
    location_hints: List[str] = Field(default_factory=list)
    environment_elements: List[EnvironmentElement] = Field(default_factory=list)

class FeatureToken(BaseModel):
    group: str
    key: str
    value: str
    subject_ref: Optional[str] = None
    location: Optional[str] = None
    evidence: Optional[str] = None

class ImageAuditRecord(BaseModel):
    image_id: Optional[str] = None
    source: Optional[str] = None

    scene_summary: List[str] = Field(default_factory=list)
    atmosphere: Atmosphere = Atmosphere()
    setting: Setting = Setting()
    people: List[Person] = Field(default_factory=list)
    objects: List[ObjectItem] = Field(default_factory=list)
    legible_text: List[str] = Field(default_factory=list)
    scripts_detected: List[ScriptInfo] = Field(default_factory=list)
    actions_relationships: List[str] = Field(default_factory=list)
    composition_lighting: CompositionLighting = CompositionLighting()
    safety: Safety = Safety()
    uncertainty: Uncertainty = Uncertainty()

    source_model_hint: Optional[str] = None
    feature_tokens: List[FeatureToken] = Field(default_factory=list)

# Optional helper (used by tool when describing from file)
def file_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "application/octet-stream"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"