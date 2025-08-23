from crewai import Task
from ..schemas.description import ImageAuditRecord

# Prompt removed by request — paste your locked guide here
STRUCTURING_GUIDE = """
You may receive an image reference.

Inputs:
- image_path (optional): {image_path}

Tool use:
- If image_path is provided, call 'DescribeImageFromFile' and treat its output as the single source of truth. Do not drop details.

Output:
- Return ONLY a JSON object that matches `ImageAuditRecord` (strict Pydantic). No prose.

-------------------------------
GENERAL PRINCIPLES
-------------------------------
• Fidelity first: preserve every observable fact from the tool description.
• Normalization: convert all free text that represents a categorical choice into the closest ENUM value defined by the schema. If you cannot safely map, use 'unknown' (or 'ambiguous' where applicable).
• Evidence-led inference: conclude sensitive attributes only when supported by direct or converging evidence; otherwise use 'unknown' or 'ambiguous'. Always attach brief evidence snippets and a confidence value when you do conclude.

-------------------------------
ENUM & FIELD COMPLIANCE
-------------------------------
• Use only declared enums for:
  - age_bucket, gender_presentation, skin_tone_label, height_bucket, body_build, body_position,
    orientation, handedness, head_covering, race_ethnicity_label, religion_label, art_style,
    position_horizontal, position_depth, BoolUnknown, Presence, ColorBasic.
• Never pass through raw phrases (e.g., number ranges, color shades, colloquialisms).
  Map to the nearest enum; otherwise 'unknown' (or 'ambiguous' for race_ethnicity_label).
• Skin tone: prefer mst_skin_tone (1..10) when possible; also set the coarse skin_tone_label. If unsure, leave mst_skin_tone unset and use 'unknown' for the label.
• Colors: map any shade/synonym to the ColorBasic enum. Unmappable → 'unknown'.

-------------------------------
OBJECTS, SYMBOLS, TEXT
-------------------------------
• Promote prominent items (tools, props, symbols, signage, flags/emblems/seals) into `objects[]` with names, attributes, colors, and positions when observable.
• For national or institutional symbols, store a canonical type label in the object attributes when determinable, otherwise 'unknown'.
• For any legible text or script-like marking, add a ScriptInfo entry (ISO-15924 code if determinable) with a short evidence snippet.

-------------------------------
SENSITIVE ATTRIBUTES (EVIDENCE-LED)
-------------------------------
• Race/Ethnicity: conclude a label only when there is explicit text, distinctive context, or multiple converging non-facial cues. If cues are weak or conflicting, prefer 'ambiguous'. You may infer from appearance or skin tone.
• Religion: conclude only when explicit symbols, attire, settings, or text support it; otherwise 'unknown'.
• Disability: conclude only from visible aids/signage; otherwise omit/unknown.
• When you conclude any sensitive label, add SensitiveEvidence with brief quotes/snippets and a reasonable confidence.

-------------------------------
FEATURE TOKENS (COUNTABLE)
-------------------------------
• Create compact FeatureTokens for small, repeatable facts (objects present/absent, key colors of important items, person-level enums, symbols, scripts, composition basics, etc.).
• Use consistent keys so repeated patterns can be counted across images.
• Include subject_ref for person-specific tokens when applicable.

-------------------------------
UNCERTAINTY
-------------------------------
• Add short notes for any non-trivial mapping or inference, especially where evidence is partial or inferred.

-------------------------------
STRICT OUTPUT
-------------------------------
Return one valid JSON object matching `ImageAuditRecord`. No extra keys. No commentary.
"""


def build_describe_image_task(agent) -> Task:
    return Task(
        description=STRUCTURING_GUIDE,
        agent=agent,
        expected_output="A single valid JSON object matching ImageAuditRecord.",
        output_pydantic=ImageAuditRecord,
    )