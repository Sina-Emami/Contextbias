"""Shared helpers for semantic token normalisation and aggregation."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .frequency_cleaner import SemanticNormalizer


def clean_token_counts(tokens: Dict[str, int]) -> Dict[str, int]:
    """Normalise and merge token counts using the semantic normaliser."""
    normalizer = SemanticNormalizer()
    observed: List[Tuple[str, int]] = []
    for token, count in tokens.items():
        if isinstance(count, (int, float)):
            weight = int(round(count))
        else:
            weight = 1
        if weight <= 0:
            weight = 1
        token_str = token if isinstance(token, str) else str(token)
        normalizer.normalise(token_str, weight)
        observed.append((token_str, weight))

    cleaned: Dict[str, int] = {}
    for token_str, weight in observed:
        norm = normalizer.normalise(token_str, weight)
        if not norm:
            continue
        cleaned[norm] = cleaned.get(norm, 0) + weight
    return dict(sorted(cleaned.items()))

