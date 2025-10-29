"""
End-to-end orchestration for the data processing pipeline.

Running this module will:

1. Count per-prompt frequencies from description JSON files.
2. Clean and normalise the prompt frequency outputs.
3. Aggregate the cleaned frequencies by role and emit role-level artefacts.
4. Merge the role-level results into a global dataset summary.

Usage
=====

    python -m decription_pipeline.data_processing.pipeline

"""

import argparse
import logging
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

from .frequency_cleaner import FrequencyCleanResult, clean_dataset
from .frequency_counter import FrequencyCounterResult, compute_frequencies
from .global_frequency_aggregator import aggregate_all_roles
from .role_frequency_aggregator import aggregate_roles

logger = logging.getLogger(__name__)


class PipelineResult(NamedTuple):
    frequency: FrequencyCounterResult
    cleaning: FrequencyCleanResult
    roles: dict
    global_summary: dict


def run_pipeline(dataset_root: Path) -> PipelineResult:
    dataset_root = dataset_root.resolve()
    logger.info("Starting frequency counting for dataset at %s", dataset_root)
    frequency_result = compute_frequencies(dataset_root)
    logger.info(
        "Frequency counting complete: %d prompt(s) visited, %d updated, %d new image(s) processed.",
        frequency_result.prompts_visited,
        frequency_result.prompts_updated,
        frequency_result.new_images,
    )

    logger.info("Normalising prompt frequencies.")
    cleaning_result = clean_dataset(dataset_root)
    logger.info(
        "Frequency cleaning complete: %d / %d prompt(s) cleaned.",
        cleaning_result.prompts_cleaned,
        cleaning_result.prompts_visited,
    )

    logger.info("Aggregating cleaned frequencies by role.")
    role_payload = aggregate_roles(dataset_root)
    logger.info("Role aggregation complete: %d role(s) processed.", len(role_payload))

    logger.info("Building global roll-up from role outputs.")
    global_payload = aggregate_all_roles(dataset_root)
    if global_payload:
        logger.info(
            "Global aggregation complete: %d categories consolidated.",
            sum(
                1
                for key, value in global_payload.items()
                if key not in {"_meta", "totals"} and isinstance(value, dict)
            ),
        )
    else:
        logger.info("Global aggregation produced no data (no role outputs found).")

    return PipelineResult(
        frequency=frequency_result,
        cleaning=cleaning_result,
        roles=role_payload,
        global_summary=global_payload,
    )


def _format_summary(result: PipelineResult) -> str:
    lines = [
        f"- Frequency counting visited {result.frequency.prompts_visited} prompt(s), "
        f"updated {result.frequency.prompts_updated}, and processed {result.frequency.new_images} new image(s).",
        f"- Frequency cleaning normalised {result.cleaning.prompts_cleaned} of "
        f"{result.cleaning.prompts_visited} prompt frequency folder(s).",
        f"- Role aggregation emitted {len(result.roles)} role file(s) in dataset/role_counting.",
    ]
    meta = result.global_summary.get("_meta") if isinstance(result.global_summary, dict) else None
    if isinstance(meta, dict):
        lines.append(
            f"- Global aggregation covers {meta.get('role_count', 0)} role(s) across "
            f"{meta.get('prompt_count', 0)} prompt(s)."
        )
    elif result.global_summary:
        lines.append("- Global aggregation completed.")
    else:
        lines.append("- Global aggregation skipped (no role data available).")
    return "\n".join(lines)


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full data processing pipeline.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("./dataset"),
        help="Path to the dataset root (default: ./dataset)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    result = run_pipeline(args.dataset)
    print(_format_summary(result))


if __name__ == "__main__":
    main()

