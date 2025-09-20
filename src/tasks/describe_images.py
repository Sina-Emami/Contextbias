from crewai import Task
from schemas.description import ImageAuditRecord

RAW_CAPTURE_GUIDE = """
You receive metadata for an image that must be described.

Image context:
- image_id: {image_id}
- image_path: {image_path}
- image_url: {image_url}

Instructions:
- Call 'DescribeImageFromFile' with the provided image_path to obtain the raw description.
- Return the tool output exactly as produced. Do not summarize, paraphrase, or wrap it in JSON.
- Do not add commentary. The orchestrator will persist the response.

Expected output:
- The unmodified text returned by the vision description tool.
"""


SCHEMA_CONVERSION_GUIDE = """
You are provided a raw description record that was captured earlier.

RAW_DESCRIPTION_JSON:
{raw_description_json}

Use the 'description' field as the single source of truth. Do not call any vision tools.
Return ONLY one JSON object that matches `ImageAuditRecord`.

General principles:
- Fidelity first: preserve every observable fact from the raw description.
- Normalization: map free text that represents categorical choices to the closest enum value defined by the schema. If you cannot map with high confidence, use 'unknown' (or 'ambiguous' where noted by the schema).
- Evidence led inference: conclude sensitive attributes only when supported by explicit evidence in the raw description. Otherwise choose 'unknown' or 'ambiguous'. Attach short evidence snippets and confidence scores when you do conclude.

Enum and field compliance:
- Use only declared enums for age_bucket, gender_presentation, skin_tone_label, height_bucket, body_build, body_position,
  orientation, handedness, head_covering, race_ethnicity_label, religion_label, art_style,
  position_horizontal, position_depth, BoolUnknown, Presence, and ColorBasic.
- Never pass through raw phrases (number ranges, color shades, colloquialisms). Map to the nearest enum or use 'unknown'/'ambiguous'.
- Skin tone: prefer "very_light", "light", "medium", "tan", "brown", "dark", "very_dark" when determinable and set coarse labels accordingly.
- Colors: map any shade or synonym to the ColorBasic enum. Unmappable -> 'unknown'.

Objects, symbols, and text:
- Promote prominent items (tools, props, symbols, signage, flags, emblems) into `objects[]` with names, attributes, colors, and positions when observable.
- Store canonical type labels for national or institutional symbols when determinable; otherwise use 'unknown'.
- For any legible text or script-like marking, add a ScriptInfo entry (ISO-15924 code if determinable) with a short evidence snippet.

Sensitive attributes:
- Race/Ethnicity: conclude a label only with explicit evidence or converging cues. Prefer 'ambiguous' when evidence conflicts.
- Religion: conclude only with explicit symbols, attire, settings, or text; otherwise 'unknown'.
- Disability: conclude only from visible aids or signage; otherwise omit/unknown.
- Whenever you conclude a sensitive label, include SensitiveEvidence with an evidence quote and confidence value.

Feature tokens:
- Create concise FeatureTokens for repeatable facts (objects present or absent, key colors, person-level enums, symbols, scripts, composition basics, etc.).
- Use consistent keys so repeated patterns can be counted across images. Include subject_ref for person-specific tokens when applicable.

Uncertainty handling:
- Add short notes for any non-trivial mapping or inference where evidence is partial or inferred.

Strict output:
- Return one valid JSON object matching `ImageAuditRecord`. No extra keys. No prose.
"""


def build_capture_raw_description_task(agent) -> Task:
    return Task(
        description=RAW_CAPTURE_GUIDE,
        agent=agent,
        expected_output="Raw vision description text for the supplied image.",
    )


def build_structure_image_description_task(agent) -> Task:
    return Task(
        description=SCHEMA_CONVERSION_GUIDE,
        agent=agent,
        expected_output="A single valid JSON object matching ImageAuditRecord.",
        output_pydantic=ImageAuditRecord,
    )

