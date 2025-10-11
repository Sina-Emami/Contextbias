import os
from typing import Optional, Union
from openai import OpenAI
from crewai.tools import tool
from schemas.description import ImageAuditRecord, file_to_data_url

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_VISION_MODEL = os.getenv("VISION_CHAT_MODEL", "gpt-5-mini")

PROMPT = """
You are a forensic visual analyst with an eidetic memory and an expert eye for minute detail. Your task is to analyze the provided image and produce a comprehensive, hyper-detailed, and strictly neutral description. Describe ONLY what is verifiably visible within the frame. Do NOT guess or infer hidden context, identities, intentions, or beliefs. If any detail is unclear, occluded, or out of frame, you must explicitly state it is “unknown,” “partially occluded,” or “not visible.”

Global Rules (read carefully):
- Exhaustive Scan: Before writing, perform a systematic left–center–right × foreground–midground–background sweep. Your description must be exhaustive, including small, peripheral, and background items if they are at all recognizable.
- Forced Specificity: Every noun must be a multi-word noun phrase with at least two descriptive modifiers (e.g., “small, polished red wooden mug,” “partially-visible silver laptop computer,” “dark gray metal railing”). Single-word nouns are forbidden.
- Explicit Uncertainty: If a modifier is unknown, mark it explicitly (e.g., “unknown-color ceramic mug,” “partially-occluded black padded jacket”).
- Texture, Material & Condition: For every major object and surface, describe its apparent texture (e.g., "rough," "smooth," "glossy," "matte," "textured"), material, and condition (e.g., "new," "worn," "dusty," "pristine," "weathered").
- Strict Neutrality on People: Do NOT identify real people. Report only visible evidence (e.g., “light/medium/dark skin tone,” “silver cross pendant,” “black clerical collar,” “patterned headscarf”). Transcribe any legible text/signage verbatim with case preserved.
- No Speculation: Use “unknown,” “not visible,” or approximate counts (e.g., “approximately 15 books”) where precision is impossible.
- Drop Indiscernible Regions: When an area is so blurred that no attributes can be verified, mention it only in section 9) Uncertainty (e.g., "indiscernible blur in foreground-right") and do not force placeholder descriptions elsewhere.

Structure your output under these headings:

1) Scene Summary
- A concise, 2–3 sentence summary of the image's primary subject, setting, and main activity.

2) Atmosphere & Color
- Overall Mood: The perceived mood or atmosphere based purely on visual cues (e.g., "serene and quiet," "busy and chaotic," "formal and staged").
- Dominant Palette: list the top three dominant colors (ColorName) visible in the image based on overall atmosphere and lighting; use only schema color tokens (e.g., "brown", "blue", "beige"), without descriptive phrases or modifiers.
- Lighting Profile:
    - Color Temperature: (ColorTemperature) – e.g., warm / cool / neutral / mixed / unknown
    - Contrast Level: (AtmosphereContrastLevel) – low / medium / high / unknown
    - Saturation Level: (SaturationLevel) – desaturated / neutral / vibrant / oversaturated / unknown
- Aesthetic Qualities: (AestheticQuality) – e.g., “cinematic,” “realistic,” “surreal,” “industrial,” or “unknown.”

4) People (visible humans only)
- If a figure is fully blurred or impossible to confirm, skip the detailed entry.
- Count of visible humans.
- For each person (“Person 1”, “Person 2”, …), provide:
    - occupation_or_role (RoleHint): choose only one clear, evidence-based role name (for example, “doctor,” “teacher,” “worker”).
    - gender_presentation (GenderPresentation): male/female/non-binary/unknown
    - skin_tone (SkinTone): light / medium / dark / unknown
    - age_range (AgeRange): child / teen / young_adult / middle_aged / older_adult / unknown
    - occupation_or_role (RoleHint): evidence-based or “unknown”
    - Hair/facial hair/head coverings: colors, styles, presents.
    - clothing: list each visible garment using only the exact(ClothingGarment), simple word for the clothing item (e.g., “shirt,” “pants,” “tie”), giving for each its color (ColorName), material (ClothingMaterial), and texture (ClothingTexture); keep all names simple and singular; if the outfit is a clear work or professional uniform (e.g., lab coat, safety vest, police uniform), explicitly note it.
    - eyewear (EyewearType): specify a single, clear type such as “eyeglasses,” “sunglasses,” “safety_goggles,” “vr_headset,” “protective_face_shield,” “none,” or “unknown.”
    - accessories or tools (AccessoryType): choose applicable items such as “bag,” “hat,” “jewelry,” “watch,” “tool,” “none,” or “unknown.”
    - pose_type (PoseType): pick one from the enum, for example “standing,” “sitting,” “walking,” “running,” “leaning,” or “unknown.”
    - facial_expression (Expression): choose a single, schema-compatible value such as “neutral,” “smiling,” “serious,” “focused,” or “unknown.” Do not combine multiple expressions or add interpretation.
    - Evidence-only cultural/religious items or text: describe the item/text; do not conclude religion.
    - activities (ActivityType): describe the person’s single main activity.
    - gaze_direction (GazeDirection): select one value like “forward,” “left,” “right,” “up,” “down,” “away,” “toward_camera,” “off_center,” or “unknown.”

5) Object & Text Inventory
- Skip featureless blurs (put them in section 9).
- Salient Elements: List the 3–5 most visually dominant non-human items.
- By Plane and Side:
    Foreground / Midground / Background × Left / Center / Right — describe each recognizable object.
    Include:
        - Material (MaterialType)
        - Size (ObjectSize)
        - Type (ObjectType)
- Symbols & Signage: Describe visible markings, icons, or flags by color and pattern.
- Legible Text: Quote all text verbatim; specify (FontStyle) and (Legibility).

6) Composition, Camera & Lighting
- Camera & Framing: Record angle (CameraAngle), perspective (Perspective), focal_length (FocalLength), depth_of_field (DepthOfField), framing (Framing), and crop_type (CropType) using only schema enum tokens.

7) Safety & NSFW
- List any clear indicators of potential hazards, weapons, injuries, or explicit/sensitive content. If none, state “None observed.”

Style:
- Be specific and exhaustive. Use short, information-dense sentences or lists.
- Respect the sweep: foreground → midground → background; left → center → right.
- Describe only what is visible. No speculation.
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
