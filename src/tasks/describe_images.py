from crewai import Task
from schemas.description import ImageAuditRecord

RAW_CAPTURE_GUIDE = """
You receive metadata for an image that must be described.

Image context:
- image_id: {image_id}
- image_path: {image_path}

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

Task:
- Use ONLY the `description` field inside RAW_DESCRIPTION_JSON. Treat it as ground truth.
- Produce a JSON object matching `ImageAuditRecord` from schemas.description (new structured schema).
- Preserve evidence fidelity: everything you assert must be supported by the raw description.
- When information is missing or unclear, use the explicit 'unknown' tokens provided by the schema.

Key mapping reminders:
1. Top-level values
   - image_id and image_path must be copied from Stage 1 metadata if present.
   - scene.summary is optional; include a concise human-readable sentence when the description supports it.

2. Atmosphere
   - mood and dominant_palette are token arrays; prefer lower_snake_case tokens.
   - lighting_profile requires enumerated values (warm/neutral/cool/unknown, etc.). Use 'unknown' when the description does not specify.

3. Environment
   - location_type is a short noun phrase (e.g., "server_room", "office"). Default to None if unclear.
   - indoor_outdoor must be: indoor, outdoor, or unknown.
   - surfaces.walls/floor/ceiling are arrays. Each entry must use token lists for material/texture/color/finish/condition.
   - spatial_layout.depth/openness/aisle_width should be short tokens ("shallow", "open", "narrow", "unknown").

4. People
   - Assign stable IDs like "P1", "P2" in order of appearance.
   - gender_presentation: female, male, nonbinary, or unknown.
   - skin_tone: light, medium, dark, or unknown.
   - age_range: child, teen, young_adult, middle_aged, older_adult, or unknown.
   - role_hint is free text.
   - hair/facial_hair/eyewear/head_covering present field must be yes/no/unknown.
   - clothing entries should capture garment_type plus color/material/texture tokens when stated.
   - pose, activities, framing details go into the respective arrays.

5. Objects
   - Provide IDs like "O1", "O2". Keep ordering consistent with the description.
   - plane: foreground/midground/background/unknown. side: left/center/right/unknown.
   - attributes.colors/material/texture/finish/condition/state should use short tokens.
   - quantity.exact is an integer when precise counts exist; otherwise leave None and populate approx (e.g., "several").

6. Background layout
   - Use object IDs inside foreground/midground/background buckets keyed by left/center/right.
   - If placement is unknown, leave arrays empty.

7. Texts, camera, lighting, safety, uncertainty
   - Text font_style should be tokenized (e.g., ["sans", "bold"]). Use ["unknown"] when unspecified.
   - Camera angle must be high/eye_level/low/unknown.
   - Lighting color_temperature/contrast_level/saturation_level follow the enumerations; default to "unknown" if absent.
   - Lighting sources capture type/count/directionality/hardness when available.
   - List hazards/nsfw indicators; use ["none"] if the description confirms absence, otherwise default to [].
   - uncertainty should collect explicit mentions of occlusions, unknowns, or ambiguous observations.

Formatting requirements:
- Output must be valid JSON with double quotes and no trailing comments.
- Ensure every list is present even when empty (schema default_factory covers this, but do not omit required keys).
- Do NOT invent facts. Prefer "unknown" or empty arrays when the raw description lacks evidence.
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

