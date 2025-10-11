import argparse
import asyncio
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence
from dotenv import load_dotenv

from analysis.schema_counts import FrequencyCounterConfig, run_counts

from utils.fs import init_scenario_root
from crew import build_image_description_crew, build_summary_report_crew
from tools.vision_description_tool import validate_image_audit_record
from pydantic import ValidationError

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


IMAGE_CONCURRENCY = _env_int("IMAGE_CONCURRENCY", 1)
ROLE_CONCURRENCY = _env_int("ROLE_CONCURRENCY", 1)
@dataclass(frozen=True)
class DescriptionJob:
    record_index: int
    image_id: str
    image_path: Path
    output_path: Path
    source_model: str


@dataclass
class PromptPlan:
    prompt_dir: Path
    role_dir: Path
    paths: dict[str, Any]
    status: str
    start_image_id: str | None
    reason: str
    last_completed_file: Path | None


def _parse_description_output(result) -> ImageAuditRecord:
    if isinstance(result, ImageAuditRecord):
        return result
    if hasattr(result, "pydantic") and isinstance(result.pydantic, ImageAuditRecord):
        return result.pydantic  # type: ignore[return-value]
    try:
        return validate_image_audit_record(result)
    except ValidationError:
        raise
    except TypeError:
        pass
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
    images_dir = root / "images"

    for path in (descriptions_dir, images_dir):
        path.mkdir(parents=True, exist_ok=True)

    paths = {
        "root": root,
        "dataset_root": root,
        "prompt_root": root,
        "descriptions": descriptions_dir,
        "images": images_dir,
        "manifest_path": root / "manifest.test.json",
        "images_info_path": images_dir / "images_info.json",
    }

    print(f"[Init] Test run configured. Outputs will be stored under: {root}")
    print(f"       Using descriptions source from: {raw_json_path}")
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

    descriptions_dir = prompt_dir / "descriptions"
    descriptions_dir.mkdir(parents=True, exist_ok=True)

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
        "descriptions": descriptions_dir,
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


def _format_prompt_label(prompt_dir: Path, dataset_root: Path) -> str:
    try:
        return str(prompt_dir.relative_to(dataset_root))
    except ValueError:
        return str(prompt_dir)


def _load_expected_ids(paths: dict[str, Any]) -> list[str]:
    info_path = Path(paths["images_info_path"])
    try:
        records = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    expected: list[str] = []
    seen: set[str] = set()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        image_id = rec.get("id") or rec.get("image_id") or rec.get("filename")
        if isinstance(image_id, str) and image_id and image_id not in seen:
            expected.append(image_id)
            seen.add(image_id)
    return expected


def assess_prompt_directory(prompt_dir: Path, dataset_root: Path) -> PromptPlan:
    paths = setup_prompt_paths(prompt_dir, dataset_root)
    expected_ids = _load_expected_ids(paths)
    role_dir = prompt_dir.parent

    descriptions_dir = Path(paths["descriptions"])

    structured_existing = {
        image_id: descriptions_dir / f"{image_id}.json"
        for image_id in expected_ids
        if (descriptions_dir / f"{image_id}.json").exists()
    }
    structured_missing = [
        image_id
        for image_id in expected_ids
        if image_id not in structured_existing
    ]

    summary_dir = descriptions_dir / "summary"
    counts_path = summary_dir / "counts.json"
    summary_path = summary_dir / "summary_report.json"

    structured_complete = not structured_missing and bool(expected_ids)
    counts_complete = counts_path.exists()
    summary_complete = summary_path.exists()

    last_completed_file: Path | None = None
    for image_id in reversed(expected_ids):
        candidate = structured_existing.get(image_id)
        if candidate:
            last_completed_file = candidate
            break

    if not expected_ids:
        return PromptPlan(
            prompt_dir=prompt_dir,
            role_dir=role_dir,
            paths=paths,
            status="complete",
            start_image_id=None,
            reason="No images found in manifest; nothing to process.",
            last_completed_file=None,
        )

    if structured_complete and counts_complete and summary_complete:
        return PromptPlan(
            prompt_dir=prompt_dir,
            role_dir=role_dir,
            paths=paths,
            status="complete",
            start_image_id=None,
            reason="All outputs already generated (structured descriptions, counts, summary).",
            last_completed_file=summary_path if summary_path.exists() else last_completed_file,
        )

    if structured_missing:
        missing_count = len(structured_missing)
        reason = f"Missing structured descriptions for {missing_count} image(s)."
        status = "pending" if not structured_existing else "resume"
        return PromptPlan(
            prompt_dir=prompt_dir,
            role_dir=role_dir,
            paths=paths,
            status=status,
            start_image_id=structured_missing[0],
            reason=reason,
            last_completed_file=last_completed_file,
        )

    missing_artifacts: list[str] = []
    if not counts_complete:
        missing_artifacts.append("counts.json")
    if not summary_complete:
        missing_artifacts.append("summary report")
    reason = "Missing " + " and ".join(missing_artifacts)
    return PromptPlan(
        prompt_dir=prompt_dir,
        role_dir=role_dir,
        paths=paths,
        status="resume",
        start_image_id=None,
        reason=reason + ".",
        last_completed_file=last_completed_file,
    )


def process_prompt_directory(
    prompt_dir: Path,
    dataset_root: Path,
    start_image_id: str | None = None,
    precomputed_paths: dict[str, Any] | None = None,
) -> None:
    paths = precomputed_paths or setup_prompt_paths(prompt_dir, dataset_root)

    label = paths.get("scenario_label") or str(prompt_dir)
    scenario_id = paths.get("scenario_id")
    if scenario_id:
        print(f"\n=== Processing prompt: {label} [{scenario_id}] ===")
    else:
        print(f"\n=== Processing prompt: {label} ===")

    describe_images(paths, start_image_id=start_image_id)

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




async def _describe_images_async(
    paths: dict,
    start_image_id: str | None = None,
) -> None:
    """Call the unified image describer crew and persist schema-compliant JSON."""
    descriptions_dir: Path = paths["descriptions"]
    descriptions_dir.mkdir(parents=True, exist_ok=True)

    info_json: Path = paths["images_info_path"]
    if not info_json.exists():
        print("No images_info.json found - nothing to describe.")
        return

    try:
        records = json.loads(info_json.read_text(encoding="utf-8"))
    except Exception:
        records = []

    if not isinstance(records, list) or not records:
        print("images_info.json is empty - nothing to describe.")
        return

    dataset_root = Path(paths.get("dataset_root") or paths.get("root") or descriptions_dir.parent)
    prompt_root = Path(paths.get("prompt_root") or paths.get("images") or descriptions_dir)
    images_dir = Path(paths.get("images") or prompt_root)

    pending_jobs: list[DescriptionJob] = []
    started = start_image_id is None

    for idx, rec in enumerate(records, start=1):
        if not isinstance(rec, dict):
            continue

        image_id = rec.get("id") or rec.get("image_id")
        if not isinstance(image_id, str) or not image_id:
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
        source_model = rec.get("source_model") or rec.get("model") or "unknown"

        image_path: Path | None = None
        if isinstance(relpath, str) and relpath:
            image_path = (dataset_root / relpath).resolve()
        elif isinstance(abspath, str) and abspath:
            image_path = Path(abspath).resolve()
        elif isinstance(filename, str) and filename:
            image_path = (images_dir / filename).resolve()

        if not image_path:
            print(f"Skipping record missing path: {rec}")
            continue
        if not image_path.exists():
            print(f"Skipping {image_id}: file not found -> {image_path}")
            continue

        output_path = descriptions_dir / f"{image_id}.json"
        if output_path.exists() and start_image_id is None:
            print(f"[Describe] Skipping {image_id}: structured description exists -> {output_path}")
            continue

        pending_jobs.append(
            DescriptionJob(
                record_index=idx,
                image_id=image_id,
                image_path=image_path,
                output_path=output_path,
                source_model=source_model,
            )
        )

    if not pending_jobs:
        return

    semaphore = asyncio.Semaphore(IMAGE_CONCURRENCY)

    async def process_job(job: DescriptionJob) -> None:
        async with semaphore:
            print(f"[Describe][start] {job.image_id} (record {job.record_index}/{len(pending_jobs)})")

            def run_job() -> str:
                crew = build_image_description_crew()
                result = crew.kickoff(
                    inputs={
                        "image_id": job.image_id,
                        "image_path": str(job.image_path),
                        "dataset_root": str(dataset_root),
                        "source_model": job.source_model,
                    }
                )
                record = _parse_description_output(result)
                if record.image.image_id.strip() == "":
                    record.image.image_id = job.image_id
                try:
                    rel = job.image_path.resolve().relative_to(dataset_root)
                    record.image.file_path = rel.as_posix()
                except Exception:
                    if record.image.file_path.strip() == "":
                        record.image.file_path = str(job.image_path)
                if not getattr(record.image, "source_model", ""):
                    record.image.source_model = job.source_model
                return record.model_dump_json(indent=2)

            desc_json = await asyncio.to_thread(run_job)
            job.output_path.write_text(desc_json, encoding="utf-8")
            print(f"   -> Structured description saved -> {job.output_path}")

    await asyncio.gather(*(process_job(job) for job in pending_jobs))


def describe_images(paths: dict, start_image_id: str | None = None) -> None:
    asyncio.run(_describe_images_async(paths, start_image_id=start_image_id))

def describe_images(paths: dict, start_image_id: str | None = None) -> None:
    asyncio.run(_describe_images_async(paths, start_image_id=start_image_id))

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




async def _execute_prompt_plans(dataset_root: Path, plans: Sequence[PromptPlan]) -> None:
    if not plans:
        return

    roles: dict[Path, list[PromptPlan]] = {}
    for plan in plans:
        roles.setdefault(plan.role_dir, []).append(plan)

    for lst in roles.values():
        lst.sort(key=lambda p: p.prompt_dir)

    semaphore = asyncio.Semaphore(ROLE_CONCURRENCY)

    async def run_role(role_dir: Path, role_plans: list[PromptPlan]) -> None:
        async with semaphore:
            label = f"role: {role_dir.name}"
            print(f"[Execute] {label} -> starting (sequential prompts, synchronous images)")
            for plan in role_plans:
                if plan.status == "complete":
                    continue
                plabel = _format_prompt_label(plan.prompt_dir, dataset_root)
                print(f"   [Prompt] {plabel} -> {plan.status}")
                await asyncio.to_thread(
                    process_prompt_directory,
                    plan.prompt_dir,
                    dataset_root,
                    plan.start_image_id,
                    plan.paths,
                )

    ordered_roles = sorted(roles.items(), key=lambda kv: kv[0])
    await asyncio.gather(*(run_role(role_dir, role_plans) for role_dir, role_plans in ordered_roles))

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

    if args.prompt:
        prompt_dirs = []
        for value in args.prompt:
            candidate = Path(value)
            candidate = (dataset_root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
            if not candidate.is_dir():
                print(f"Skipping prompt path (missing directory): {value}")
                continue
            prompt_dirs.append(candidate)
    else:
        prompt_dirs = discover_prompt_directories(dataset_root)

    if not prompt_dirs:
        print(f"No prompt directories discovered under {dataset_root}.")
        return

    unique_prompts: list[Path] = []
    seen: set[Path] = set()
    for prompt_dir in prompt_dirs:
        resolved = prompt_dir.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_prompts.append(resolved)

    prompt_dirs = sorted(unique_prompts)

    if args.limit is not None and args.limit >= 0:
        prompt_dirs = prompt_dirs[: args.limit]

    plans: list[PromptPlan] = []
    for prompt_dir in prompt_dirs:
        try:
            plan = assess_prompt_directory(prompt_dir, dataset_root)
        except Exception as exc:
            label = _format_prompt_label(prompt_dir, dataset_root)
            print(f"[Error] Failed to assess {label}: {exc}")
            continue
        plans.append(plan)

    if not plans:
        print("No actionable prompt directories after assessment.")
        return

    work_plans: list[PromptPlan] = []
    for plan in plans:
        label = _format_prompt_label(plan.prompt_dir, dataset_root)
        if plan.status == "complete":
            print(f"[Skip] {label} already fully processed; skipping.")
            continue
        if plan.status == "pending":
            print(f"[Start] {label} -> starting fresh. {plan.reason}")
            if plan.start_image_id:
                print(f"         First image id: {plan.start_image_id}")
        else:
            resume_msg = plan.reason
            if plan.last_completed_file:
                try:
                    display_file = plan.last_completed_file.relative_to(dataset_root)
                except ValueError:
                    display_file = plan.last_completed_file
                print(
                    f"[Resume] {label} -> {resume_msg} Resuming from last generated file: {display_file}"
                )
            else:
                print(f"[Resume] {label} -> {resume_msg}")
            if plan.start_image_id:
                print(f"         Next image id: {plan.start_image_id}")
        work_plans.append(plan)

    if not work_plans:
        print("All prompt folders are fully processed. Nothing to do.")
        return

    print(f"Queued {len(work_plans)} prompt folder(s) for processing.")
    asyncio.run(_execute_prompt_plans(dataset_root, work_plans))


if __name__ == "__main__":
    main()




