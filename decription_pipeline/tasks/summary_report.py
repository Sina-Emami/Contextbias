"""Task definitions for summarizing aggregated count JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from crewai import Task

SUMMARY_REPORT_GUIDE = """
You receive aggregated value-count data produced from an image auditing schema.

SUMMARY_COUNTS_JSON:
{summary_json}

Instructions:
1. Treat each top-level category (cohort) independently. Never merge observations across cohorts.
2. Equivalence-only grouping inside a cohort:
   - Merge tokens only when they clearly represent the same concept (e.g., spelling variants, singular/plural, accents).
   -Use semantic similarity and context to confirm equivalence; do not rely on hard-coded synonym lists.
   - If two tokens can co-exist in the dataset (e.g., distinct enumerated values), they must be reported as separate groups.
   - Enumerated/categorical dimensions must yield one group per distinct value observed; issue separate groups such as key="gender_presentation", sub_key="male" and key="gender_presentation", sub_key="female".
3. Database-friendly identifiers:
   - Represent each group with two explicit identifiers: "key" (dimension/schema field) and "sub_key" (canonical value), both lower_snake_case.
   - The members array should normally contain exactly one token (normalized form of sub_key). Only include additional entries when they normalize to the exact same token after removing case/spacing/diacritics.
   - Keep "canonical_label" human-readable but distinct from key/sub_key. Do NOT concatenate them (avoid patterns like "age_child").
   - Example pattern: "blue_palette" for shades of blue, "upper_body_clothing" for garments covering torso.
4. Attribute handling occurs at the cohort level only. Provide distributions for attributes such as color, material, pattern, texture, size, finish, etc. When helpful, map shades or variants into generalized tokens (e.g., "blue_palette"). Do not duplicate attribute stats inside groups.
5. Compute normalized shares (0-1 floats) alongside raw counts for both groups and cohort-level attributes.
6. Remove empty, unknown, or zero-count items from the final JSON.
7. Document the grouping methodology in metadata so the process is auditable and extensible.
8. Output must be valid JSON (UTF-8) suitable for statistical evaluation and storage in relational or vector databases.

Required JSON structure:
{{
  "metadata": {{
    "source_summary_version": "1.1",
    "num_images": <int>,
    "grouping_methodology": <text>,
    "cohort_labels": [<string tokens>]
  }},
  "cohorts": [
    {{
      "cohort": <identifier>,
      "groups": [
        {
          "key": <dimension token>,
          "sub_key": <canonical value token>,
          "canonical_label": <display label>,
          "total_count": <int>,
          "normalized_share": <float>,
          "members": [<tokens merged into this group>]
        }}
      ],
      "attributes": {{
        "color": {{"value_counts": {{...}}, "normalized": {{...}} }},
        "material": {{...}},
        ... (include only attributes that exist for the cohort)
      }},
      "notes": <optional insight or omit>
    }}
  ],
  "salient_observations": [
    {{
      "topic": <token>,
      "insight": <summary>,
      "evidence_groups": [<references to cohort/key/sub_key combinations>]
    }
  ]
}}
Do not emit markdown. Return JSON only.
"""


def build_summary_report_task(agent, counts_path: Path | str) -> Task:
    """Create a task that feeds the counts JSON into the reporting agent."""
    path = Path(counts_path)
    if not path.exists():
        raise FileNotFoundError(f"Counts JSON not found: {path}")

    data: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    description = SUMMARY_REPORT_GUIDE.format(summary_json=payload)

    return Task(
        description=description,
        agent=agent,
        expected_output=(
            "Valid JSON with cohorts, per-group key/sub_key identifiers, precise equivalence-only groupings, and cohort-level attribute distributions."
        ),
    )


__all__ = ["build_summary_report_task"]