import os
import json
from pathlib import Path
from dotenv import load_dotenv

from analysis.schema_counts import FrequencyCounterConfig, run_counts

from utils.fs import init_scenario_root
from crew import (
    build_raw_description_crew,
    build_structured_description_crew,
    build_bias_ingest_crew,
    build_bias_reasoning_crew,
    build_summary_report_crew,
)
from schemas.description import ImageAuditRecord
from schemas.bias import BiasReport

load_dotenv()

#  Hardcoded inputs per your request
SCENARIO_ID = "Tech_SCN001"


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


def capture_raw_descriptions(paths: dict) -> list[dict]:
    """Stage 1: Call the raw description crew for each image and store results."""
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
    raw_records: list[dict] = []
    seen_ids: set[str] = set()
    for entry in _load_json_array(raw_records_path):
        if not isinstance(entry, dict):
            continue
        image_id = entry.get("image_id")
        if image_id and image_id in seen_ids:
            continue
        if image_id:
            seen_ids.add(image_id)
        raw_records.append({
            "image_id": image_id,
            "image_path": entry.get("image_path"),
            "description": entry.get("description"),
        })

    raw_crew = build_raw_description_crew()

    for idx, rec in enumerate(records, start=1):
        image_id = rec.get("id")
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

        if not image_id or not image_path:
            print(f"Skipping record missing id/path: {rec}")
            continue
        if not image_path.exists():
            print(f"Skipping {image_id}: file not found -> {image_path}")
            continue

        print(f"[Stage 1] Capturing raw description {idx}/{len(records)} - {image_id} (local: {image_path})")
        raw_result = raw_crew.kickoff(inputs={
            "image_id": image_id,
            "image_path": str(image_path),
        })
        raw_text = _extract_raw_description_output(raw_result)
        sanitized = {
            "image_id": image_id,
            "image_path": str(image_path),
            "description": raw_text,
        }
        raw_records = [entry for entry in raw_records if entry.get("image_id") != image_id]
        raw_records.append(sanitized)
        raw_records_path.write_text(json.dumps(raw_records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   -> Raw description stored -> {raw_records_path}")

    return raw_records


def structure_descriptions(paths: dict) -> None:
    """Stage 2: Convert raw descriptions into structured ImageAuditRecord JSON."""
    descriptions_dir: Path = paths["descriptions"]
    descriptions_dir.mkdir(parents=True, exist_ok=True)

    raw_dir: Path = paths.get("raw_descriptions") or (descriptions_dir / "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_records_path = raw_dir / "raw_descriptions.json"

    raw_records = _load_json_array(raw_records_path)
    if not raw_records:
        print("No raw descriptions available - skipping structured conversion.")
        return

    structured_crew = build_structured_description_crew()

    for idx, entry in enumerate(raw_records, start=1):
        if not isinstance(entry, dict):
            continue
        image_id = entry.get("image_id")
        image_path = entry.get("image_path")
        if not image_id or not image_path:
            print(f"Skipping raw entry missing id/path: {entry}")
            continue

        print(f"[Stage 2] Structuring description {idx}/{len(raw_records)} - {image_id}")
        payload = json.dumps({
            "image_id": image_id,
            "image_path": image_path,
            "description": entry.get("description"),
        }, indent=2, ensure_ascii=False)
        structured_result = structured_crew.kickoff(inputs={
            "image_id": image_id,
            "image_path": image_path,
            "raw_description_json": payload,
        })
        desc = _parse_description_output(structured_result)
        if getattr(desc, "image_id", None) is None:
            desc.image_id = image_id
        if getattr(desc, "image_path", None) is None:
            desc.image_path = image_path
        out_path = descriptions_dir / f"{image_id}.json"
        out_path.write_text(desc.model_dump_json(indent=2), encoding="utf-8")
        print(f"   -> Structured description saved -> {out_path}")


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


def _parse_bias_report_output(result) -> BiasReport | dict | str:
    if isinstance(result, BiasReport):
        return result
    if hasattr(result, "pydantic") and isinstance(result.pydantic, BiasReport):
        return result.pydantic  # type: ignore[return-value]
    if hasattr(result, "json_dict") and result.json_dict:
        return result.json_dict  # type: ignore[return-value]
    if hasattr(result, "raw") and result.raw:
        try:
            return json.loads(result.raw)  # type: ignore[attr-defined]
        except Exception:
            return result.raw  # type: ignore[attr-defined]
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except Exception:
            return result
    return str(result)


def run_analyze_bias(paths: dict) -> None:
    """Step 3:
    - Ingest each description JSON one-by-one with a memory-enabled agent (idempotent upsert).
    - Finalize repetition summary to biases/repeat_summary_full.json
    - Reason over the summary with Replicate OSS 20B to produce biases/bias_report.json
    """
    descriptions_dir: Path = paths["descriptions"]
    biases_dir: Path = paths["biases"]
    biases_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(descriptions_dir.glob("*.json"))
    if not files:
        print("No descriptions found - skipping bias analysis.")
        return

    state_path = biases_dir / "agg_state.json"
    out_path = biases_dir / "repeat_summary_full.json"

    # 3a) Ingest sequentially (agent has memory=True)
    ingest_crew = build_bias_ingest_crew(files, state_path, out_path)
    summary_result = ingest_crew.kickoff()

    # Try to load the finalized summary from disk (authoritative)
    if out_path.exists():
        rep_summary = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        # Fallback to crew output
        if hasattr(summary_result, "json_dict") and summary_result.json_dict:
            rep_summary = summary_result.json_dict  # type: ignore[attr-defined]
        elif hasattr(summary_result, "raw") and summary_result.raw:
            try:
                rep_summary = json.loads(summary_result.raw)  # type: ignore[attr-defined]
            except Exception:
                rep_summary = {}
        else:
            rep_summary = {}

    # 3b) Reason to BiasReport (Replicate OSS 20B)
    reason_crew = build_bias_reasoning_crew()
    reason_out = reason_crew.kickoff(inputs={
        "summary_json": json.dumps(rep_summary, ensure_ascii=False),
        "extra_context": f"Scenario ID: {SCENARIO_ID}",
    })
    report = _parse_bias_report_output(reason_out)

    # Save BiasReport
    if isinstance(report, BiasReport):
        (biases_dir / "bias_report.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
    elif isinstance(report, dict):
        (biases_dir / "bias_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        (biases_dir / "bias_report_raw.txt").write_text(str(report), encoding="utf-8")

    print(f"?? Bias summary  -> {out_path}")
    print(f"?? Bias report   -> {biases_dir / 'bias_report.json'} (or bias_report_raw.txt)")


if __name__ == "__main__":
    paths = setup_scenario(SCENARIO, SCENARIO_ID)
    # capture_raw_descriptions(paths)
    # structure_descriptions(paths)
    # counts_path = summarize_description_counts(paths)
    run_summary_report(paths)
    # run_analyze_bias(paths)
