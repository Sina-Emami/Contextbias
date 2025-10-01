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
- Reference the enum literal groups defined in schemas.description (Mood, DominantPalette, LightingColorTemperature, etc.) whenever you populate categorical fields so every value traces back to a declared enumeration.
- Never hallucinate or infer details. When the description omits a field, output "unknown" (or ["unknown"] for list-based enums) instead of guessing.
- Always use schema enumerations verbatim; do not invent new tokens or alter casing.
- If the raw description calls an element completely blurry or impossible to identify, omit that element, add a short explanation to scene.uncertainty, and keep associated categorical fields as "unknown".

Key mapping reminders:
1. Top-level values
   - image_id and image_path must be copied from Stage 1 metadata if present.
   - scene.summary is optional; include a concise human-readable sentence only when the description supports it.

2. Atmosphere
   - mood uses the Mood enum tokens (calm, tense, joyful, melancholic, mysterious, romantic, foreboding, whimsical, minimalistic, dramatic). Use ["unknown"] if unspecified.
   - dominant_palette uses DominantPalette tokens (monochrome, analogous, complementary, triadic, tetradic, earth_tones, neon, pastel, high_key, low_key). Use ["unknown"] when unclear.
   - lighting_profile values map to LightingColorTemperature, AtmosphereContrastLevel, SaturationLevel, and AestheticQualities. Set each to "unknown" or ["unknown"] when the description lacks evidence.

3. Environment
   - location_type is a short noun phrase (e.g., "server_room", "office"). Leave null if unclear.
   - indoor_outdoor must be indoor, outdoor, or unknown.
   - time_of_day_hint uses TimeOfDayHint tokens (dawn, morning, noon, afternoon, dusk, evening, night, midnight, unknown).
   - weather uses Weather tokens (clear, sunny, cloudy, overcast, rain, drizzle, storm, snow, fog, mist, windy, hail, unknown).
   - surfaces.walls/floor/ceiling are arrays. Each entry must use token lists for material/texture/color/finish/condition.
   - spatial_layout.depth/openness/aisle_width should be short tokens ("shallow", "open", "narrow", etc.) with "unknown" when not described.

4. People
   - Skip individuals the description marks as fully blurry or unidentifiable; instead add a note to scene.uncertainty and leave population-level fields as "unknown".
   - Assign stable IDs like "P1", "P2" in order of appearance.
   - gender_presentation: female, male, nonbinary, or unknown.
   - skin_tone: light, medium, dark, or unknown.
   - age_range: child, teen, young_adult, middle_aged, older_adult, or unknown.
   - role_hint is free text; leave null rather than fabricate details.
   - hair.present must be yes/no/unknown. Hair style tokens come from the Hairstyle enum; hair color tokens come from HairColor. Default to ["unknown"] when absent.
   - facial_hair/eyewear/head_covering present fields must be yes/no/unknown with supporting tokens only when stated.
   - face_emotion must use the FaceEmotion enum, defaulting to "unknown".
   - gaze_direction must use the GazeDirection enum, defaulting to "unknown".
   - occlusions must use the OcclusionState enum, defaulting to "unknown".
   - clothing entries should capture garment_type plus ClothesColor/material/texture tokens when stated. Use ["unknown"] for color when unspecified.
   - pose and activities remain free-form token lists grounded in the description.

5. Objects
   - Skip object entries when the description explicitly says they are indiscernible blobs or fully blurred; log that ambiguity under scene.uncertainty with associated tokens left as "unknown".
   - Provide IDs like "O1", "O2". Keep ordering consistent with the description.
   - plane: foreground/midground/background/unknown. side: left/center/right/unknown.
   - attributes.colors/material/texture/finish/condition/state should use evidence-backed tokens; prefer "unknown" to speculation.
   - quantity.exact is an integer when precise counts exist; otherwise leave None and populate approx (e.g., "several").

6. Background layout
   - Use object IDs inside foreground/midground/background buckets keyed by left/center/right.
   - If placement is unknown, leave arrays empty rather than inventing coordinates.

7. Texts, camera, lighting, safety, uncertainty
   - Text font_style should be tokenized (e.g., ["sans", "bold"]). Use ["unknown"] when unspecified.
   - Camera angle uses the CameraAngle enum; perspective uses Perspective; focal_length uses FocalLength; depth_of_field uses DepthOfField; framing uses Framing. Mark missing values as "unknown" or ["unknown"].
   - Lighting color_temperature uses LightingColorTemperature; contrast_level uses ContrastLevel; saturation_level uses SaturationLevel.
   - Lighting sources capture type (SourcesType), count (SourcesCount), directionality (Directionality), and hardness (Hardness). Shadows must use the Shadows enum. Use "unknown"/["unknown"] when the description lacks evidence.
   - List hazards/nsfw indicators; use ["none"] if the description confirms absence, otherwise default to [].
   - uncertainty should collect explicit mentions of occlusions, unknowns, ambiguous observations, and any elements you deliberately omitted for being fully indiscernible.

Formatting requirements:
- Output must be valid JSON with double quotes and no trailing comments.
- Ensure every list is present even when empty (schema defaults cover this, but do not omit required keys).
- Do NOT invent facts. Prefer "unknown" or ["unknown"] when the raw description lacks evidence.
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


