import os
from typing import Optional, Union
from openai import OpenAI
from crewai.tools import tool
from schemas.description import ImageAuditRecord, file_to_data_url

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_VISION_MODEL = os.getenv("VISION_CHAT_MODEL", "gpt-5-mini")

PROMPT = """
You are a careful vision assistant.
Analyze the image and extract only the fields required by the schema below.
Return plain text following exactly the sections and keys provided.
Do not rename keys or add sections.
Use enum tokens verbatim where applicable.
When evidence is missing, use "unknown" or [].
Do not use markdown, bullet symbols, or code fences in your response.

+ Global Rules
    - Avoid redundancy: Deduplicate lists and use concise phrasing.
    - No speculation: Use "unknown" or [] where precision is impossible.
    - Drop indiscernible regions: If an area is too blurred to analyze, mark "unknown" or [].
    - Describe only what is visually verifiable. Never infer hidden context, identities, or intentions.
    - Focus on observable evidence.

+ SceneAppearance
Define the overall look, lighting, and mood of the scene based only on visual cues.
    - Mood: Choose one from ("calm", "tense", "joyful", "melancholic", "mysterious", "romantic", "foreboding", "whimsical", "minimalistic", "dramatic", "unknown").
    - Dominant Color: List the top three dominant colors (ColorName) visible in the image.
    - Color Temperature: Choose one (ColorTemperature): "warm", "cool", "neutral", "mixed", or "unknown".
    - Aesthetic Qualities: Choose one (AestheticQuality): "minimal", "baroque", "retro", "surreal", "abstract", "realistic", "fantasy", "cinematic", "gritty", "dreamy", "industrial", "unknown".
    - Weather: Describe visible atmospheric conditions using one of the "clear", "sunny", "cloudy", "overcast", "rain", "drizzle", "storm", "snow", "fog", "mist", "windy", "hail", "unknown".

Schema Section:
    SceneAppearance:
    - mood: <Mood>
    - color_temperature: <ColorTemperature>
    - aesthetic_quality: <AestheticQuality>
    - dominant_colors: [<ColorName>, <ColorName>, <ColorName>]
    - weather: <Weather>

+ Camera
Describe the viewpoint, focus depth, and framing style as they appear in the image — how the scene is captured rather than what is captured.
    - Perspective: ("first_person", "third_person", "birdseye", "wormseye", "eye_level", "high_angle", "unknown")
    - Depth of Field: ("shallow", "medium", "deep", "infinite", "unknown")
    - Framing: ("tight", "medium", "wide", "balanced", "asymmetric", "unknown")

Schema Section:
    Camera:
    - depth_of_field: <DepthOfField>
    - framing: <Framing>
    - perspective: <Perspective>

+ People (Visible Humans Only)
Describe each visibly distinct person in the image.
If a figure is too blurred or unclear to identify key traits, skip that person.

For each person (“Person 1”, “Person 2”, …), record:
    - gender_presentation (GenderPresentation): male / female / non-binary / unknown
    - skin_tone (SkinTone): light / medium / dark / unknown
    - age_range (AgeRange): child / teen / young_adult / middle_aged / older_adult / unknown
    - Hair, facial hair, tattoos, and head coverings: include colors, styles, presence indicators, and note tattoo visibility.
    - Clothing: list each visible garment using simple names (e.g., “shirt,” “pants,” “tie”) with its color(s); if clearly a work or professional uniform, note it explicitly.
    - head_covering_type: list zero or more HeadCoveringType values ("hat","helmet", "hood", "hijab", "turban", "veil", "headband", "beanie", "hairnet", "none", "unknown").
    - Eyewear: specify one clear EyewearType (“eyeglasses”, “sunglasses”, etc.).
    - Accessories: choose visible AccessoryType (“bag”, “hat”, “watch”, etc.).
    - Pose: pick one PoseType ("standing", "sitting", "kneeling", "lying", "leaning", "bending", "arms_crossed", "hands_on_hips", "unknown").
    - Expression: choose one Expression (“neutral”, “smiling”, “focused”, etc.).
    - Activity: describe the person’s main visible action in 1–3 words.
    - Gaze Direction: select a GazeDirection (“forward”, “left”, “toward_camera”, etc.).
    - Body Type: slim / average / athletic / heavyset / unknown

Schema Section:
    People:
    - persons:
    - person_id: <short token or unknown>
        accessories: [<AccessoryType>, ...]
        activities: [<short free text tokens>, ...]
        age_range: <AgeRange>
        body_type: <BodyType>
        clothing:
        - clothing_garment: <short free text>
            clothing_color: [<ColorName>, ...]
        eyewear_present: <yes|no|unknown>
        eyewear_type: <EyewearType>
        facial_hair_present: <yes|no|unknown>
        facial_hair_style: <FacialHairStyle>
        facial_hair_color: <ColorName>
        gender_presentation: <GenderPresentation>
        gaze_direction: <GazeDirection>
        hair_present: <yes|no|unknown>
        hair_color: <HairColor>
        hair_style: <HairStyle>
        tattoo_present: <yes|no|unknown>
        head_covering_present: <yes|no|unknown>
        head_covering_type: <HeadCoveringType>
        pose: <PoseType>
        expression: <Expression>
        role_hint: <short free text or unknown>
        skin_tone: <SkinTone>

+ Objects
List the three most visually dominant non-human items(tools, furniture, vehicles, etc.), especially those associated with the people in the image.
Use short common names, deduplicate near-duplicates, and stop once you have three confident entries.
If fewer than three objects are clearly identifiable, list only the ones you can confirm.

Schema Section:
    Objects:
    - items: [<object_name>, <object_name>, <object_name>]

+ Totals
Provide simple totals for visible people and objects.

Schema Section:
    Totals:
    - images: 1
    - object_instances: <int|unknown>
    - people_instances: <int|unknown>

+ Enum Guidance (Strict Usage)

Use only valid schema tokens for all enumerated fields:
Mood, ColorTemperature, AestheticQuality, ColorName, Weather, DepthOfField, Framing, Perspective, AccessoryType, AgeRange, BodyType, GenderPresentation, GazeDirection, HairStyle, HairColor, EyewearType, FacialHairStyle, HeadCoveringType, Expression, PoseType.

For free-text lists (activities, clothing, object names), use short, evidence-based terms.
If unclear, output [].

Never infer hidden context, purpose, or identity — describe visible evidence only.
"""


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def validate_image_audit_record(payload: Union[str, dict, ImageAuditRecord]) -> ImageAuditRecord:
    """Normalize and validate a payload as an ImageAuditRecord."""
    if isinstance(payload, ImageAuditRecord):
        return payload
    if isinstance(payload, dict):
        return ImageAuditRecord.model_validate(payload)
    if isinstance(payload, str):
        return ImageAuditRecord.model_validate_json(_strip_code_fences(payload))
    if hasattr(payload, "model_dump"):
        return ImageAuditRecord.model_validate(payload.model_dump())  # type: ignore[call-arg]
    if hasattr(payload, "dict"):
        return ImageAuditRecord.model_validate(payload.dict())  # type: ignore[call-arg]
    if hasattr(payload, "raw"):
        return ImageAuditRecord.model_validate_json(_strip_code_fences(payload.raw))  # type: ignore[attr-defined]
    raise TypeError(f"Unsupported payload type for validation: {type(payload)!r}")


def _describe_from_source(image_url: str, prompt: Optional[str] = None) -> str:
    resp = _client.chat.completions.create(
        model=_VISION_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful vision assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (prompt or PROMPT)},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
    )
    return resp.choices[0].message.content.strip()


@tool("DescribeImageFromFile")
def describe_image_from_file_tool(path: str, prompt: Optional[str] = None) -> str:
    """Read a local image file, convert it to a data: URL, and return a detailed, neutral description."""
    data_url = file_to_data_url(path)
    return _describe_from_source(data_url, prompt)

__all__ = ["describe_image_from_file_tool", "validate_image_audit_record"]
