"""Agent responsible for interpreting aggregated counts and writing a structured report."""
from __future__ import annotations

import os
from typing import Iterable

from crewai import Agent

# Cohort hints are informational only � they make prompts deterministic without coupling to data.
COHORT_HINTS: tuple[str, ...] = (
    "people.demographics",
    "people.appearance",
    "people.clothing",
    "people.pose_activity",
    "people.positions",
    "objects",
    "texts",
    "environment",
    "lighting",
    "camera",
    "atmosphere",
    "safety",
    "uncertainty",
)


def _csv(values: Iterable[str]) -> str:
    return ", ".join(sorted({v for v in values if v}))


def build_summary_report_agent() -> Agent:
    """Create an agent that groups only equivalent tokens and emits database-ready JSON."""
    llm = os.getenv("SUMMARY_REPORT_LLM", "gpt-5-mini")
    return Agent(
        name="Schema Cohort Analyst",
        role=(
            "Interpret aggregated schema counts, preserve category separation, and summarize them"
            " into reliable cohorts with clean statistical metadata."
            " Always separate distinct enumerated values so downstream statistics stay faithful."
        ),
        goal=(
            "Cluster only tokens that represent the same underlying concept, keep attributes at the"
            " category level, and produce normalized JSON suited for analytics and vector storage."
        ),
        backstory=(
            "Experienced data modeler who designs fact tables for multimodal bias studies. Skilled at"
            " spotting near-duplicate terminology without collapsing distinct concepts, and errs on the"
            " side of one token per value unless evidence proves equivalence."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=1,
        memory=False,
        additional_info={
            "cohort_hints": _csv(COHORT_HINTS),
        },
    )


__all__ = ["build_summary_report_agent"]
