from crewai import Task
from schemas.description import ImageAuditRecord

DESCRIBE_IMAGE_GUIDE = """
You receive metadata for an image that must be described and structured.

Inputs:
- image_id: {image_id}
- image_path: {image_path}
- dataset_root: {dataset_root}
- source_model: {source_model}

Tooling:
- You MUST call the tool 'DescribeImageFromFile' with the given image_path to inspect the image content. Treat the tool's text as evidence only; do not copy its formatting.

Task:
- Emit exactly one JSON object that conforms to schemas.description.ImageAuditRecord.
- Use only schema enum tokens where defined; when evidence is missing, write "unknown" for scalars and [] for lists.
- Do not add or rename keys. Keep arrays present even if empty.
- Set image.image_id to {image_id}.
- Set image.file_path to the path of the image relative to {dataset_root} (use forward slashes). If you cannot compute the relative path, use {image_path} as-is.
- Set image.source_model to {source_model}.

Cohorts:
- scene_appearance: mood, color_temperature, contrast_level, aesthetic_qualities[], dominant_colors[], weather.
- camera: depth_of_field, framing, perspective.
- people: provide `persons` as an array. Each person entry must include person_id (short token or "unknown"), accessories[], activities[], age_range, body_type, clothing (array of { clothing_garment, clothing_color[] }), eyewear_present, eyewear_type, facial_hair_present, facial_hair_style, facial_hair_color, gender_presentation, gaze_direction, hair_present, hair_color, hair_style, tattoo_present, head_covering_present, head_covering_type, pose, expression, role_hint, skin_tone. Deduplicate list values within each person.
- objects: `items` mapping where each object name maps to { size, color[], material[] }.
- safety: hazards, nsfw (short free text or "unknown").
- totals: images=1, object_instances, people_instances (integers when explicit, otherwise "unknown").

People aggregation:
- When multiple people are present, create one entry per person in `people.persons`. Use consistent identifiers if the tool output references them.

Formatting:
- Return VALID JSON only (no explanations, no markdown).
"""


def build_describe_image_task(agent) -> Task:
    return Task(
        description=DESCRIBE_IMAGE_GUIDE,
        agent=agent,
        expected_output="A single valid JSON object matching ImageAuditRecord.",
        output_pydantic=ImageAuditRecord,
    )
