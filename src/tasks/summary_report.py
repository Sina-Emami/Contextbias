"""Task definitions for summarizing aggregated count JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crewai import Task

SUMMARY_REPORT_GUIDE = """
You receive a JSON payload that contains value-count summaries extracted from structured image descriptions.

SUMMARY_COUNTS_JSON:\n{summary_json}\n
Instructions:
1. Parse the JSON and reason about the relative importance of each category (people, objects, environment, lighting, etc.).
2. Merge tokens that express the same or closely-related concepts (e.g., inflections, synonyms, near-duplicates) by using semantic similarity and context found inside the counts. Describe the grouping approach in general terms, without listing hard-coded examples.
3. For every grouping you create, provide the unified label, the member tokens that were merged, the combined count, and any notable supporting metrics.
4. Highlight the categories with the strongest signals or imbalances as separate observations.
5. Remove any keys, arrays, or objects that are empty, contain only null/unknown/"", or whose aggregate counts fall to zero after grouping.
6. Return ONLY valid JSON with the following top-level structure:
   {{
     "metadata": {{
       "source_summary_version": "1.1",
       "num_images": <int>,
       "grouping_methodology": <short text describing the general aggregation technique>
     }},
     "groupings": [
       {{
         "category": <high-level area>,
         "group_label": <name assigned to the merged concept>,
         "members": [<list of original tokens>],
         "aggregate_count": <int>,
         "share_of_category": <float 0-1 showing normalized share if meaningful>,
         "notes": <concise supporting detail>
       }}
     ],
     "salient_observations": [
       {{
         "topic": <short identifier>,
         "insight": <what the counts reveal>,
         "evidence": <list of JSON-pointer-like references to supporting groupings>
       }}
     ]
   }}
Ensure that arrays omit entries that would otherwise be empty, and that optional keys are entirely omitted when no value is available. Do not output markdown; only minified or pretty JSON is acceptable.
"""


def build_summary_report_task(agent, counts_path: Path | str) -> Task:
    """Create a task that feeds the counts JSON into the reporting agent."""
    path = Path(counts_path)
    if not path.exists():
        raise FileNotFoundError(f"Counts JSON not found: {path}")
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    description = SUMMARY_REPORT_GUIDE.format(summary_json=payload)
    return Task(
        description=description,
        agent=agent,
        expected_output="Valid JSON report with metadata, groupings, and salient_observations.",
    )


__all__ = ["build_summary_report_task"]
