import os
from typing import Optional
from openai import OpenAI
from crewai.tools import tool
from schemas.description import file_to_data_url

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_VISION_MODEL = os.getenv("VISION_CHAT_MODEL", "gpt-5-mini")

PROMPT = """
You are a meticulous visual describer. Analyze the image and produce a comprehensive, neutral description. Describe ONLY what is clearly visible. Do NOT guess or infer hidden context, identities, or beliefs. If something is unclear or not in frame, say “unknown” or “not visible”.

Global rules (read carefully):
- Exhaustive scan: perform a left–center–right × foreground–midground–background sweep before writing. Include small/background items if they are recognizable. If an item is too small to identify, label it generically (e.g., “small unknown object”) with visible attributes (color/material/shape) and its position.
- Every noun is a multi-word noun phrase with at least one modifier (e.g., “red wooden mug”, “blue-and-white striped flag”, “dark metal railing”). Avoid single-word nouns.
- If a modifier is unknown, mark it explicitly (e.g., “unknown-color ceramic mug”, “partially-occluded black jacket”).
- Do NOT identify real people or assign race/ethnicity/religion. Report only visible evidence (e.g., “light/medium/dark skin tone”, “silver cross pendant”, “black clerical collar”, “patterned headscarf”). Transcribe any legible text/signage verbatim with case preserved.
- No speculation. Use “unknown”, “not visible”, “partially occluded”, or approximate counts (e.g., “≈12”) when appropriate.

Structure your output under these headings:

1) Scene Summary
- 2–3 sentences summarizing who/what/where and the main activity.

2) Atmosphere & Color
- Overall mood/atmosphere.
- Dominant color palette (top 3–5 multi-word color phrases), overall color temperature (warm/cool/neutral), contrast (low/medium/high), saturation (muted/neutral/vivid).
- Notable aesthetics (e.g., “soft natural lighting”, “high-contrast studio lighting”, “fine grain low-light snapshot”).

3) Setting & Environment
- Indoor/outdoor, location hints, weather (if applicable).
- Environment elements as multi-word phrases (e.g., “light gray concrete sidewalk”, “polished wooden table surface”).
- Spatial layout: explicitly note placements using both plane and side (e.g., “foreground-left”, “midground-center”, “background-right”).
- If shelves/walls/displays contain many items, provide approximate counts and summaries (e.g., “≈30 assorted hardcover books with multicolor spines on background-left white shelving”).

4) People (visible humans only)
- Count of visible humans.
- For each person (Person 1, Person 2, …):
  - Apparent age range wording (e.g., “appears middle-aged”), perceived gender presentation (“appears …”), visible skin tone (neutral: light/medium/dark), hair/facial hair/eyewear/head coverings.
  - Clothing as multi-word items (e.g., “dark blue denim jacket”, “white cotton t-shirt with small black logo”).
  - Posture and facial expression as multi-word phrases.
  - Evidence-only cultural/religious items or text (e.g., “silver cross pendant”, “patterned headscarf”, “text on badge reading ‘Pastor’”). Do NOT conclude religion.
  - Apparent role only if strongly indicated by visible cues; mark uncertainty if unsure.

5) Objects, Animals & Text
- Salient non-human elements, each with: name, visible color(s), material(s), quantity, and position (e.g., “two red plastic traffic cones in foreground-left”).
- Perform an object inventory by plane:
  - Foreground inventory (left/center/right): list all recognizable items.
  - Midground inventory (left/center/right): list all recognizable items.
  - Background inventory (left/center/right): list all recognizable items, including small items (e.g., books, frames, switches, light fixtures, plants, signs).
- Symbols/emblems/flags/seals: describe their patterns/colors/materials and placement (e.g., “blue field with white stars on vertical flag at background-left”).
- Transcribe legible text: quote exact text and give location (e.g., “white sans-serif sign reading ‘OPEN’ above center doorway”). If partially legible, mark unclear parts with “(illegible)” or “[?]”.

6) Actions & Relationships
- What each person is doing and their interactions with objects/others, written as concise, multi-word phrases. Include gaze direction and object handling when visible.

7) Composition, Camera & Lighting
- Camera angle, focal-length feel (wide/normal/telephoto), depth of field, framing/cropping, leading lines/symmetry.
- Light sources, directionality, hardness/softness, and shadows as multi-word phrases. Note reflections, glare, and specular highlights if present.

8) Safety & NSFW
- Any safety/NSFW indicators (weapons, injuries, explicit content). If none, say “none observed”.

9) Uncertainty
- Bullet any ambiguous or partially occluded elements, approximate counts, or places where details could not be read (e.g., “text partially occluded on background-right plaque”, “flag design partly out of frame”, “small dark object on table not identifiable”).

Style:
- Be specific and exhaustive. Use short, information-dense sentences or lists.
- Respect the sweep: foreground → midground → background; left → center → right.
- Describe only what is visible. No speculation.
"""


def _describe_from_url(image_url: str, prompt: Optional[str] = None) -> str:
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


@tool("DescribeImageFromURL")
def describe_image_from_url_tool(image_url: str, prompt: Optional[str] = None) -> str:
    """Return a detailed, neutral description of the image at the given URL (including data: URLs)."""
    return _describe_from_url(image_url, prompt)


@tool("DescribeImageFromFile")
def describe_image_from_file_tool(path: str, prompt: Optional[str] = None) -> str:
    """Read a local image file, convert it to a data: URL, and return a detailed, neutral description."""
    data_url = file_to_data_url(path)
    return _describe_from_url(data_url, prompt)