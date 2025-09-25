import argparse
import asyncio
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from dotenv import load_dotenv

from analysis.schema_counts import FrequencyCounterConfig, run_counts

from utils.fs import init_scenario_root
from crew import (
    build_raw_description_crew,
    build_structured_description_crew,
    build_summary_report_crew,
)
from schemas.description import ImageAuditRecord

load_dotenv()

# Root folder containing the dataset prompt directories
DEFAULT_DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "dataset")).resolve()


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        parsed = int(value)
        return parsed if parsed >= 1 else default
    except ValueError:
        return default


RAW_STAGE_CONCURRENCY = _env_int("RAW_STAGE_CONCURRENCY", 10)
STRUCT_STAGE_CONCURRENCY = _env_int("STRUCT_STAGE_CONCURRENCY", 10)


@dataclass
class ResumeState:
    last_file: Path | None
    start_prompt: Path | None
    start_image_id: str | None


@dataclass(frozen=True)
class RawDescriptionJob:
    record_index: int
    image_id: str
    image_path: Path


@dataclass(frozen=True)
class StructuredDescriptionJob:
    record_index: int
    image_id: str
    image_path: str
    payload: str
    output_path: Path


def _extract_raw_description_output(result) -> str:
    if isinstance(result, str):
        return result.strip()
    if hasattr(result, "raw") and isinstance(result.raw, str) and result.raw.strip():
        return result.raw.strip()
    if hasattr(result, "output") and isinstance(result.output, str) and result.output.strip():
        return result.output.strip()
    if hasattr(result, "json_dict") and isinstance(result.json_dict, dict):
        description = result.json_dict.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
    text_value = str(result).strip()
    if not text_value:
        raise RuntimeError(f"Unexpected raw description output type: {type(result)} {result}")
    return text_value


def _parse_description_output(result) -> ImageAuditRecord:
    if isinstance(result, ImageAuditRecord):
        return result
    if hasattr(result, "pydantic") and isinstance(result.pydantic, ImageAuditRecord):
        return result.pydantic  # type: ignore[return-value]
    if isinstance(result, dict):
        return ImageAuditRecord.model_validate(result)
    if hasattr(result, "model_dump"):  # pydantic v2 model
        return ImageAuditRecord.model_validate(result.model_dump())
    if hasattr(result, "dict"):        # pydantic v1 model
        return ImageAuditRecord.model_validate(result.dict())
    if isinstance(result, str):
        return ImageAuditRecord.model_validate_json(result)
    if hasattr(result, "raw"):
        return ImageAuditRecord.model_validate_json(result.raw)  # type: ignore[attr-defined]
    raise RuntimeError(f"Unexpected description output type: {type(result)} {result}")


def setup_scenario(scenario: str, scenario_id: str) -> dict:
    """Create the scenario root, subfolders, and manifest. Return the paths dict."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your environment or .env file.")
    paths = init_scenario_root(scenario_id, scenario)
    print(f"[Init] Scenario folders ready at: {paths['root']}")
    return paths


def setup_test_paths(raw_json_path: Path, output_root: Path | None = None) -> dict:
    """Prepare folders so structuring & reporting can run against a canned raw JSON."""
    raw_json_path = Path(raw_json_path).resolve()
    if not raw_json_path.exists():
        raise FileNotFoundError(f"Raw descriptions file not found: {raw_json_path}")
    if raw_json_path.is_dir():
        raise ValueError("Expected a raw descriptions JSON file, received a directory.")

    raw_dir = raw_json_path.parent
    root = Path(output_root).resolve() if output_root else raw_dir.parent
    root.mkdir(parents=True, exist_ok=True)
    descriptions_dir = root / "descriptions"
    biases_dir = root / "biases"
    images_dir = root / "images"

    for path in (descriptions_dir, biases_dir, images_dir):
        path.mkdir(parents=True, exist_ok=True)

    paths = {
        "root": root,
        "raw_descriptions": raw_dir,
        "descriptions": descriptions_dir,
        "biases": biases_dir,
        "images": images_dir,
        "manifest_path": root / "manifest.test.json",
        "images_info_path": images_dir / "images_info.json",
    }

    print(f"[Init] Test run configured. Outputs will be stored under: {root}")
    print(f"       Using raw descriptions from: {raw_json_path}")
    return paths


def _find_manifest_file(prompt_dir: Path) -> Path:
    """Locate the JSON manifest describing images within a prompt directory."""
    prompt_dir = Path(prompt_dir).resolve()
    candidates = [
        prompt_dir / "manifest.json",
        prompt_dir / "prompt" / "manifest.json",
        prompt_dir / "prompt" / "images.json",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    for candidate in prompt_dir.glob("*.json"):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            return candidate

    raise FileNotFoundError(f"No manifest JSON found in {prompt_dir}")


def _scenario_slug(parts: Sequence[str]) -> str:
    tokens: list[str] = []
    for part in parts:
        cleaned = "".join(ch if ch.isalnum() else "_" for ch in part)
        cleaned = cleaned.strip("_")
        if not cleaned:
            continue
        segments = [seg for seg in cleaned.split("_") if seg]
        if not segments:
            continue
        tokens.append("_".join(segment.lower() for segment in segments))
    return "_".join(tokens) or "scenario"


def _derive_scenario_id(dataset_root: Path, prompt_dir: Path) -> str:
    try:
        relative = prompt_dir.relative_to(dataset_root)
    except ValueError:
        relative = prompt_dir
    return _scenario_slug(relative.parts)


def _resolve_image_path(record: dict, dataset_root: Path, prompt_dir: Path) -> Path | None:
    """Best-effort resolution of an image file path for a manifest record."""
    candidates: list[Path] = []

    for key in ("abspath", "absolute_path", "image_path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            candidates.append(Path(value))

    for key in ("relpath", "relative_path"):
        value = record.get(key)
        if isinstance(value, str) and value:
            rel_candidate = Path(value)
            candidates.append((dataset_root / rel_candidate).resolve())
            candidates.append((prompt_dir / rel_candidate).resolve())
            candidates.append((prompt_dir / rel_candidate.name).resolve())

    filename = record.get("filename")
    if isinstance(filename, str) and filename:
        candidates.append((prompt_dir / filename).resolve())

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return None


def setup_prompt_paths(prompt_dir: Path, dataset_root: Path | None = None) -> dict:
    """Prepare per-prompt folders within the dataset for pipeline outputs."""
    dataset_root = Path(dataset_root or DEFAULT_DATASET_ROOT).resolve()
    prompt_dir = Path(prompt_dir).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not prompt_dir.exists() or not prompt_dir.is_dir():
        raise FileNotFoundError(f"Prompt directory missing: {prompt_dir}")

    manifest_path = _find_manifest_file(prompt_dir)
    try:
        manifest_records = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest JSON is invalid: {manifest_path}") from exc

    if not isinstance(manifest_records, list):
        raise ValueError(f"Expected manifest JSON array at {manifest_path}")

    raw_dir = prompt_dir / "raw_descriptions"
    descriptions_dir = prompt_dir / "descriptions"
    biases_dir = prompt_dir / "biases"

    for folder in (raw_dir, descriptions_dir, biases_dir):
        folder.mkdir(parents=True, exist_ok=True)

    normalized_records: list[dict] = []
    for entry in manifest_records:
        if not isinstance(entry, dict):
            continue
        normalized = dict(entry)
        normalized.pop("abspath", None)
        resolved = _resolve_image_path(normalized, dataset_root, prompt_dir)
        if resolved:
            try:
                normalized["relpath"] = str(resolved.relative_to(dataset_root))
            except ValueError:
                pass
        else:
            print(f"   !! Unable to resolve image path for record {entry.get('id')} in {prompt_dir}")
        normalized_records.append(normalized)

    images_info_path = prompt_dir / "images_info.json"
    images_info_path.write_text(
        json.dumps(normalized_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    scenario_id = _derive_scenario_id(dataset_root, prompt_dir)
    try:
        relative_prompt = prompt_dir.relative_to(dataset_root)
        scenario_label = f"Dataset prompt: {'/'.join(relative_prompt.parts)}"
    except ValueError:
        scenario_label = f"Dataset prompt: {prompt_dir}"

    return {
        "root": dataset_root,
        "dataset_root": dataset_root,
        "prompt_root": prompt_dir,
        "images": prompt_dir,
        "raw_descriptions": raw_dir,
        "descriptions": descriptions_dir,
        "biases": biases_dir,
        "images_info_path": images_info_path,
        "manifest_path": manifest_path,
        "scenario_id": scenario_id,
        "scenario_label": scenario_label,
    }


def discover_prompt_directories(dataset_root: Path) -> list[Path]:
    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.exists():
        return []
    manifests = sorted(dataset_root.rglob("manifest.json"))
    prompt_dirs = [manifest.parent for manifest in manifests if manifest.parent.is_dir()]
    return prompt_dirs


def determine_resume_state(dataset_root: Path, prompt_dirs: Sequence[Path]) -> ResumeState:
    dataset_root = Path(dataset_root).resolve()
    last_file: Path | None = None

    for prompt_dir in sorted({Path(p).resolve() for p in prompt_dirs}):
        try:
            manifest_path = _find_manifest_file(prompt_dir)
        except FileNotFoundError:
            continue

        try:
            manifest_records = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if not isinstance(manifest_records, list):
            continue

        descriptions_dir = prompt_dir / "descriptions"
        for record in manifest_records:
            if not isinstance(record, dict):
                continue
            image_id = record.get("id") or record.get("image_id")
            if not image_id:
                continue
            candidate = descriptions_dir / f"{image_id}.json"
            if candidate.exists():
                last_file = candidate.resolve()
                continue
            return ResumeState(last_file=last_file, start_prompt=prompt_dir, start_image_id=image_id)

    return ResumeState(last_file=last_file, start_prompt=None, start_image_id=None)


def process_prompt_directory(prompt_dir: Path, dataset_root: Path, start_image_id: str | None = None) -> None:
    paths = setup_prompt_paths(prompt_dir, dataset_root)

    label = paths.get("scenario_label") or str(prompt_dir)
    scenario_id = paths.get("scenario_id")
    if scenario_id:
        print(f"\n=== Processing prompt: {label} [{scenario_id}] ===")
    else:
        print(f"\n=== Processing prompt: {label} ===")

    raw_records = capture_raw_descriptions(paths, start_image_id=start_image_id)
    if not raw_records:
        print("[Stage 1] No raw descriptions captured or available to process.")

    structure_descriptions(paths, start_image_id=start_image_id)
    counts_path = summarize_description_counts(paths)
    run_summary_report(paths, counts_path=counts_path)

def _load_json_array(path: Path) -> list:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


async def _capture_raw_descriptions_async(
    paths: dict,
    start_image_id: str | None = None,
) -> list[dict]:
    """Stage 1: Call the raw description crew for each image and store results asynchronously."""
    descriptions_dir: Path = paths["descriptions"]
    raw_dir: Path = paths.get("raw_descriptions") or (descriptions_dir / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    info_json: Path = paths["images_info_path"]
    images_dir: Path = paths["images"]
    root_dir: Path = paths["root"]

    if not info_json.exists():
        print("No images_info.json found - nothing to describe.")
        return []

    try:
        records = json.loads(info_json.read_text(encoding="utf-8"))
    except Exception:
        records = []

    if not isinstance(records, list) or not records:
        print("images_info.json is empty - nothing to describe.")
        return []

    raw_records_path = raw_dir / "raw_descriptions.json"
    ordered_ids: list[str] = []
    for rec in records:
        if isinstance(rec, dict):
            image_id = rec.get("id") or rec.get("image_id")
            if image_id:
                ordered_ids.append(image_id)

    existing_entries: dict[str, dict] = {}
    for entry in _load_json_array(raw_records_path):
        if not isinstance(entry, dict):
            continue
        image_id = entry.get("image_id")
        if image_id:
            existing_entries[image_id] = {
                "image_id": image_id,
                "image_path": entry.get("image_path"),
                "description": entry.get("description"),
            }

    processed_ids: set[str] = {
        image_id
        for image_id, payload in existing_entries.items()
        if payload.get("description")
    }

    total_records = len(records)
    pending_jobs: list[RawDescriptionJob] = []
    started = start_image_id is None

    for idx, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            continue

        image_id = rec.get("id") or rec.get("image_id")
        if not image_id:
            print(f"Skipping record missing id: {rec}")
            continue

        if not started:
            if image_id == start_image_id:
                started = True
            else:
                continue

        relpath = rec.get("relpath")
        abspath = rec.get("abspath")
        filename = rec.get("filename")

        image_path: Path | None = None
        if relpath:
            image_path = (root_dir / relpath).resolve()
        elif abspath:
            image_path = Path(abspath)
        elif filename:
            image_path = (images_dir / filename).resolve()

        if not image_path:
            print(f"Skipping record missing path: {rec}")
            continue
        if not image_path.exists():
            print(f"Skipping {image_id}: file not found -> {image_path}")
            continue

        if image_id in existing_entries:
            existing_entries[image_id]["image_path"] = str(image_path)

        if image_id in processed_ids:
            print(f"[Stage 1] Skipping raw description (already exists) - {image_id}")
            continue

        pending_jobs.append(
            RawDescriptionJob(
                record_index=idx,
                image_id=image_id,
                image_path=image_path,
            )
        )

    if not pending_jobs:
        ordered_records = [
            existing_entries[item_id]
            for item_id in ordered_ids
            if item_id in existing_entries
        ]
        extra_ids = sorted(
            iid for iid in existing_entries.keys() if iid not in ordered_ids
        )
        ordered_records.extend(existing_entries[iid] for iid in extra_ids)
        return ordered_records

    semaphore = asyncio.Semaphore(RAW_STAGE_CONCURRENCY)
    write_lock = asyncio.Lock()

    async def process_job(job: RawDescriptionJob) -> None:
        async with semaphore:
            print(
                f"[Stage 1][async start] {job.image_id} (record {job.record_index}/{total_records})"
            )

            def run_job():
                crew = build_raw_description_crew()
                return crew.kickoff(
                    inputs={
                        "image_id": job.image_id,
                        "image_path": str(job.image_path),
                    }
                )

            try:
                raw_result = await asyncio.to_thread(run_job)
                raw_text = _extract_raw_description_output(raw_result)
            except Exception as exc:
                print(f"   !! Stage 1 failed for {job.image_id}: {exc}")
                raise

            sanitized = {
                "image_id": job.image_id,
                "image_path": str(job.image_path),
                "description": raw_text,
            }

            print(f"[Stage 1][async done] {job.image_id}")
            async with write_lock:
                existing_entries[job.image_id] = sanitized
                processed_ids.add(job.image_id)
                ordered_records = [
                    existing_entries[item_id]
                    for item_id in ordered_ids
                    if item_id in existing_entries
                ]
                extra_ids = sorted(
                    iid for iid in existing_entries.keys() if iid not in ordered_ids
                )
                ordered_records.extend(existing_entries[iid] for iid in extra_ids)
                raw_records_path.write_text(
                    json.dumps(ordered_records, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(f"   -> Raw description stored -> {raw_records_path}")

    await asyncio.gather(*(process_job(job) for job in pending_jobs))

    ordered_records = [
        existing_entries[item_id]
        for item_id in ordered_ids
        if item_id in existing_entries
    ]
    extra_ids = sorted(
        iid for iid in existing_entries.keys() if iid not in ordered_ids
    )
    ordered_records.extend(existing_entries[iid] for iid in extra_ids)
    return ordered_records


def capture_raw_descriptions(
    paths: dict,
    start_image_id: str | None = None,
) -> list[dict]:
    return asyncio.run(
        _capture_raw_descriptions_async(paths, start_image_id=start_image_id)
    )


async def _structure_descriptions_async(
    paths: dict, start_image_id: str | None = None
) -> None:
    """Stage 2: Convert raw descriptions into structured ImageAuditRecord JSON asynchronously."""
    descriptions_dir: Path = paths["descriptions"]
    descriptions_dir.mkdir(parents=True, exist_ok=True)

    raw_dir: Path = paths.get("raw_descriptions") or (descriptions_dir / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_records_path = raw_dir / "raw_descriptions.json"

    raw_records = _load_json_array(raw_records_path)
    if not raw_records:
        print("No raw descriptions available - skipping structured conversion.")
        return

    raw_map: dict[str, dict] = {}
    for entry in raw_records:
        if isinstance(entry, dict):
            image_id = entry.get("image_id")
            if image_id:
                raw_map[image_id] = entry

    images_info_path = Path(paths.get("images_info_path", ""))
    ordered_ids: list[str] = []
    if images_info_path.exists():
        try:
            info_records = json.loads(images_info_path.read_text(encoding="utf-8"))
            if isinstance(info_records, list):
                for rec in info_records:
                    if isinstance(rec, dict):
                        image_id = rec.get("id") or rec.get("image_id")
                        if image_id:
                            ordered_ids.append(image_id)
        except json.JSONDecodeError:
            pass

    if not ordered_ids:
        ordered_ids = list(raw_map.keys())

    pending_ids = [iid for iid in ordered_ids if iid in raw_map]
    extras = [iid for iid in raw_map.keys() if iid not in ordered_ids]
    pending_ids.extend(sorted(extras))

    if start_image_id and start_image_id in pending_ids:
        start_idx = pending_ids.index(start_image_id)
        pending_ids = pending_ids[start_idx:]

    total_pending = len(pending_ids)
    pending_jobs: list[StructuredDescriptionJob] = []

    for idx, image_id in enumerate(pending_ids, start=1):
        entry = raw_map.get(image_id)
        if not entry:
            continue

        image_path = entry.get("image_path")
        if not image_path:
            print(f"Skipping raw entry missing path: {entry}")
            continue

        out_path = descriptions_dir / f"{image_id}.json"
        if out_path.exists():
            print(f"[Stage 2] Skipping {image_id}: structured description exists -> {out_path}")
            continue

        payload = json.dumps(
            {
                "image_id": image_id,
                "image_path": image_path,
                "description": entry.get("description"),
            },
            indent=2,
            ensure_ascii=False,
        )

        pending_jobs.append(
            StructuredDescriptionJob(
                record_index=idx,
                image_id=image_id,
                image_path=image_path,
                payload=payload,
                output_path=out_path,
            )
        )

    if not pending_jobs:
        return

    semaphore = asyncio.Semaphore(STRUCT_STAGE_CONCURRENCY)
    write_lock = asyncio.Lock()

    async def process_job(job: StructuredDescriptionJob) -> None:
        async with semaphore:
            print(
                f"[Stage 2][async start] {job.image_id} (record {job.record_index}/{total_pending})"
            )

            def run_job() -> str:
                crew = build_structured_description_crew()
                result = crew.kickoff(
                    inputs={
                        "image_id": job.image_id,
                        "image_path": job.image_path,
                        "raw_description_json": job.payload,
                    }
                )
                desc = _parse_description_output(result)
                if getattr(desc, "image_id", None) is None:
                    desc.image_id = job.image_id
                if getattr(desc, "image_path", None) is None:
                    desc.image_path = job.image_path
                return desc.model_dump_json(indent=2)

            try:
                desc_json = await asyncio.to_thread(run_job)
            except Exception as exc:
                print(f"   !! Stage 2 failed for {job.image_id}: {exc}")
                raise

            print(f"[Stage 2][async done] {job.image_id}")
            async with write_lock:
                job.output_path.write_text(desc_json, encoding="utf-8")
                print(f"   -> Structured description saved -> {job.output_path}")

    await asyncio.gather(*(process_job(job) for job in pending_jobs))


def structure_descriptions(paths: dict, start_image_id: str | None = None) -> None:
    asyncio.run(
        _structure_descriptions_async(paths, start_image_id=start_image_id)
    )


def summarize_description_counts(paths: dict, output_filename: str = "counts.json") -> Path | None:
    """Aggregate structured descriptions into value-count summaries."""
    descriptions_dir: Path = paths["descriptions"]
    structured_files = sorted(p for p in descriptions_dir.glob('*.json') if p.is_file())
    if not structured_files:
        print("No structured descriptions found - skipping frequency summary.")
        return None

    summary_dir: Path = descriptions_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_path = summary_dir / output_filename

    print(f"[Stage 2.1] Aggregating structured descriptions -> {output_path}")
    config = FrequencyCounterConfig()
    try:
        run_counts(descriptions_dir, output_path, config)
    except Exception as exc:
        print(f"   !! Failed to build frequency summary: {exc}")
        raise

    print(f"   -> Frequency counts saved -> {output_path}")
    return output_path


def run_summary_report(paths: dict, counts_path: Path | None = None, output_filename: str = "summary_report.json") -> Path | None:
    """Use the summary-report crew to turn count aggregates into a structured JSON report."""
    descriptions_dir: Path = paths["descriptions"]
    summary_dir = descriptions_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    target_counts = Path(counts_path) if counts_path else summary_dir / "counts.json"
    if not target_counts.exists():
        print(f"Counts JSON missing at {target_counts} - rebuilding counts summary.")
        regenerated = summarize_description_counts(paths)
        if regenerated is None:
            print("Unable to create counts summary; skipping report generation.")
            return None
        target_counts = regenerated

    report_path = Path(output_filename) if Path(output_filename).is_absolute() else summary_dir / output_filename

    print(f"[Stage 2.2] Generating summary report -> {report_path}")
    crew = build_summary_report_crew(target_counts)
    result = crew.kickoff()

    payload = None
    if hasattr(result, "json_dict") and result.json_dict:
        payload = result.json_dict
    elif hasattr(result, "pydantic") and result.pydantic is not None:
        candidate = result.pydantic
        if hasattr(candidate, "model_dump"):
            payload = candidate.model_dump()
        elif hasattr(candidate, "dict"):
            payload = candidate.dict()
    else:
        raw_value = getattr(result, "raw", None)
        if isinstance(raw_value, str) and raw_value.strip():
            try:
                payload = json.loads(raw_value)
            except Exception:
                payload = raw_value.strip()
        elif isinstance(result, str):
            try:
                payload = json.loads(result)
            except Exception:
                payload = result

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception as exc:
            raise RuntimeError(f"Summary report output was not valid JSON: {payload}") from exc

    if payload is None:
        raise RuntimeError("Summary report agent returned no usable data.")

    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   -> Summary report saved -> {report_path}")
    return report_path



def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the bias analysis pipeline across dataset prompts.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the dataset root directory (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Relative or absolute path to a specific prompt directory (repeatable).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N prompt directories.",
    )

    args = parser.parse_args(argv)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your environment or .env file.")

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    prompt_dirs: list[Path] = []

    if args.prompt:
        for value in args.prompt:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = (dataset_root / candidate).resolve()
            else:
                candidate = candidate.resolve()
            if not candidate.is_dir():
                print(f"Skipping prompt path (missing directory): {value}")
                continue
            prompt_dirs.append(candidate)
    else:
        prompt_dirs = discover_prompt_directories(dataset_root)

    if not prompt_dirs:
        print(f"No prompt directories discovered under {dataset_root}.")
        return

    seen: set[Path] = set()
    unique_prompts: list[Path] = []
    for prompt_dir in prompt_dirs:
        resolved = prompt_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_prompts.append(resolved)

    prompt_dirs = sorted(unique_prompts)

    if args.limit is not None and args.limit >= 0:
        prompt_dirs = prompt_dirs[: args.limit]

    resume_state = determine_resume_state(dataset_root, prompt_dirs)

    if resume_state.last_file:
        try:
            relative_last = resume_state.last_file.relative_to(dataset_root)
            print(f"Resuming from last generated file: {relative_last}")
        except ValueError:
            print(f"Resuming from last generated file: {resume_state.last_file}")
    else:
        print("No previous progress found, starting from the beginning.")

    if resume_state.start_prompt is None:
        if resume_state.last_file is not None:
            print("All selected prompts already processed. Nothing new to do.")
        else:
            # No prior progress and nothing identified to process (likely empty manifests)
            print("No actionable items discovered in the selected prompts.")
        return

    processing_started = False
    for prompt_dir in prompt_dirs:
        if not processing_started:
            if prompt_dir == resume_state.start_prompt:
                processing_started = True
                start_image_id = resume_state.start_image_id
            else:
                continue
        else:
            start_image_id = None

        process_prompt_directory(prompt_dir, dataset_root, start_image_id=start_image_id)
        # After the first processing iteration, ensure subsequent prompts start fresh
        resume_state.start_prompt = None


if __name__ == "__main__":
    main()
