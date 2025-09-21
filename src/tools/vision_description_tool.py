import os
from typing import Optional
from openai import OpenAI
from crewai.tools import tool
from schemas.description import file_to_data_url

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

Structure your output under these headings:

1) Scene Summary
- A concise, 2–3 sentence summary of the image's primary subject, setting, and main activity.

2) Atmosphere & Color
- Overall Mood: The perceived mood or atmosphere based purely on visual cues (e.g., "serene and quiet," "busy and chaotic," "formal and staged").
- Dominant Palette: The top 3–5 multi-word color phrases (e.g., “deep mahogany browns,” “muted sage greens,” “off-white creams”).
- Lighting Profile: Overall color temperature (warm/cool/neutral), contrast level (low/medium/high), and color saturation (muted/neutral/vivid).
- Aesthetic Qualities: Any notable visual style (e.g., “soft, diffuse natural lighting,” “high-contrast, hard-edged studio lighting,” “low-resolution digital photo with visible artifacts”).

3) Setting & Environment
- General Location: Indoor/outdoor, time of day hints (e.g., "long shadows suggesting late afternoon"), and weather (if applicable).
- Environmental Context: Based on visible evidence, describe the likely function of the space (e.g., “appears to be a commercial office space due to cubicles,” “resembles a public library with book stacks”).
- Architectural & Surface Details: Describe the visible surfaces in detail. For walls, floors, and ceilings, specify their material, texture, color, and condition (e.g., “light-beige painted drywall with a subtle orange-peel texture,” “dark-stained, high-gloss hardwood floor with minor scuff marks,” “white acoustic ceiling tile with a grid pattern”).
- Spatial Layout: Explicitly note the placement of key elements using plane and side (e.g., “foreground-left,” “midground-center,” “background-right”). Describe the sense of depth and openness (e.g., "cramped and enclosed," "open and expansive").

4) People (visible humans only)
- Count of visible humans.
- For each person (“Person 1”, “Person 2”, …), provide:
    - Required demographics & role (must appear exactly with these keys):
    - gender_presentation: “appears …” (e.g., “appears female-presenting”) or “unknown”
    - skin_tone: “light” / “medium” / “dark” (or “unknown”)
    - age_range: e.g., “appears young adult”, “appears middle-aged” (or “unknown”)
    - occupation_or_role: evidence-based (e.g., “appears construction worker based on high-visibility safety vest and hard hat”); else “unknown”
    - Hair/facial hair/eyewear/head coverings: colors, styles, visibility.
    - Clothing & Attire: A list of all visible clothing as multi-word items, including material and fit if discernible (e.g., “loose-fitting, dark blue denim jacket,” “white, smooth cotton t-shirt with a small, black embroidered logo”).
    - Posture & facial expression: concise multi-word phrases (e.g., “upright relaxed posture”, “slight smile with neutral gaze”).
    - Evidence-only cultural/religious items or text: describe the item/text; do not conclude religion.
    - Gaze & orientation: facing direction, gaze target if visible.
    - Occlusions: what is partially hidden or cropped.

5) Object & Text Inventory
- Salient Elements: Begin by listing the 3-5 most prominent non-human elements.
- Perform an object inventory by plane:
  - Foreground (Left/Center/Right): List and describe all recognizable items with at least two modifiers, texture, and material.
  - Midground (Left/Center/Right): List and describe all recognizable items with at least two modifiers, texture, and material.
  - Background (Left/Center/Right): This section must be exceptionally detailed. Describe all visible background elements, no matter how small. Include furniture, decor (framed pictures, posters), fixtures (light switches, power outlets, thermostats), distant landscape features, items on shelves (provide approximate counts), and any other ambient objects. Describe the arrangement of these items (e.g., "neatly stacked," "randomly scattered").
- Symbols & Signage: Describe any symbols, emblems, flags, or seals by pattern, color, and placement.
- Legible Text: Quote all legible text verbatim, preserving case. Describe the font style (e.g., "white sans-serif text") and location. Mark unclear parts with “(illegible)” or “[?]”.

6) Actions & Relationships
- Ongoing Actions: Describe what each person is actively doing, including gaze direction and interactions with objects or other people (e.g., "Person 1 is looking at a silver laptop screen while typing," "Person 2 is handing a white ceramic mug to Person 3").
- Inferred Interactions: Based only on proximity, posture, and gaze, describe the apparent relationships between individuals (e.g., "Person 1 and Person 2 are positioned face-to-face, suggesting a direct conversation," "Person 3 is standing apart from the main group, looking away").

7) Composition, Camera & Lighting
- Camera & Framing: Note the camera angle (eye-level, high-angle, low-angle), estimated focal length (wide-angle, normal, telephoto), and depth of field (shallow/deep). Describe framing and cropping (e.g., "tightly cropped on the subject's face," "symmetrically framed composition").
- Lighting Analysis: Identify the number and type of apparent light sources (e.g., "a single, soft light source from the upper-left," "multiple harsh point lights"). Describe the directionality, hardness/softness, and resulting shadows (e.g., "long, soft shadows indicating a light source low and to the right"). Note any visible reflections, lens flare, or spec```

8) Safety & NSFW
- List any clear indicators of potential hazards, weapons, injuries, or explicit/sensitive content. If none, state “None observed.”

9) Uncertainty
- Bullet ambiguous or partially occluded elements, approximate counts, or places where details could not be read (e.g., “text partially occluded on background-right plaque”, “flag design partly out of frame”, “small dark object on table not identifiable”).

Style:
- Be specific and exhaustive. Use short, information-dense sentences or lists.
- Respect the sweep: foreground → midground → background; left → center → right.
- Describe only what is visible. No speculation.
"""


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