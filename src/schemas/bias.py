from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field


class BiasFinding(BaseModel):
    bias: str = Field(..., description="Short name of the potential bias")
    evidence: List[str]
    images_affected: str
    severity: str
    confidence: float = Field(..., ge=0, le=1)
    mitigation: List[str]


class BiasReport(BaseModel):
    summary_notes: List[str]
    findings: List[BiasFinding]
