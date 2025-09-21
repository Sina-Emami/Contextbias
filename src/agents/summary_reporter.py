"""Agent responsible for interpreting aggregated counts and writing a structured report."""
from __future__ import annotations

import os
from crewai import Agent


def build_summary_report_agent() -> Agent:
    """Create an agent that groups related concepts and emits a clean JSON report."""
    llm = os.getenv("SUMMARY_REPORT_LLM", "gpt-4o-mini")
    return Agent(
        name="Summary Insights Analyst",
        role="Synthesize count summaries into higher-level patterns and clean JSON findings.",
        goal=(
            "Read aggregated value-count data, cluster conceptually similar items, and respond with a"
            " machine-friendly JSON report that excludes unknown or empty fields."
        ),
        backstory=(
            "Seasoned data analyst who understands how to merge near-duplicate tokens, identify"
            " dominant themes, and structure findings for downstream automation workflows."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        memory=False,
    )


__all__ = ["build_summary_report_agent"]
