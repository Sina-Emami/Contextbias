"""Task definitions for summarizing aggregated count JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from crewai import Task

SUMMARY_REPORT_GUIDE = """
You receive aggregated value-count data produced from an image auditing schema.

SUMMARY_COUNTS_JSON:\n{summary_json}\n

Instructions:
1. Treat each top-level category (cohort) independently. Never merge observations across cohorts.
2. Within a cohort, only merge tokens when they describe the exact same concept (spelling variants,
   singular/plural, accent differences). Distinct garments, ages, skin tones, or activities must remain separate.
   Use semantic similarity and context to confirm equivalence; do not rely on hard-coded synonym lists.
3. Produce database-friendly group keys. Prefer lower_snake_case tokens that reflect the concept, not a raw member.
   Example pattern: "blue_family" for shades of blue, "upper_body_clothing" for garments covering torso.
4. Attribute handling occurs at the cohort level only. For each cohort, provide distributions for attributes such as
   color, material, pattern, texture, size, finish, or other relevant fields. Do NOT push attribute stats down to individual groups.
   Group attribute families into generalized tokens (e.g., "blue_family", "organic_materials").
5. Compute normalized shares (0-1 floats) alongside raw counts for both groups and cohort-level attributes.
6. Remove empty, unknown, or zero-count items from the final JSON.
7. Document the grouping methodology in metadata so the process is auditable and extensible.
8. Output must be valid JSON, encoded in UTF-8, suitable for statistical evaluation and storage in relational or vector databases.

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
        {{
          "group_key": <token>,
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
      "evidence_groups": [<references to cohort/group combinations>]
    }}
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
            "Valid JSON capturing cohorts, precise groupings, and cohort-level attribute distributions."
        ),
    )


__all__ = ["build_summary_report_task"]
