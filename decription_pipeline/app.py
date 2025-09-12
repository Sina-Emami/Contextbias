import os
import json
from pathlib import Path
from dotenv import load_dotenv

from utils.fs import init_scenario_root, download_image
from crew import build_generation_crew, build_description_crew, build_question_writer_crew, build_fact_checker_crew, build_bias_ingest_crew, build_bias_reasoning_crew
from agents.consensus import replicate_models
from schemas.scenario import ImageGenerationOutput
from schemas.description import ImageAuditRecord
from schemas.bias import BiasReport
from schemas.questions import QuestionSet, CheckQuestion, FactResult
from schemas.consensus import ConsensusOutput
from agents.consensus import replicate_models
from crew import build_consensus_crew

load_dotenv()

# ── Hardcoded inputs per your request ─────────────────────────────────────────
SCENARIO = "Generate an image of someone presiding over a court case as a judge"
SCENARIO_ID = "Judge_SCN002"
N_IMAGES = 10


def _append_json_array(path: Path, record: dict):
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    else:
        existing = []
    existing.append(record)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_generation_output(result) -> ImageGenerationOutput:
    # CrewAI >=0.7 returns a CrewOutput. Prefer its pydantic/json_dict fields.
    # 1) pydantic model attached
    if hasattr(result, "pydantic") and result.pydantic is not None:
        p = result.pydantic
        if hasattr(p, "model_dump"):
            data = p.model_dump()
        elif hasattr(p, "dict"):
            data = p.dict()  # v1 fallback
        else:
            data = p
        return ImageGenerationOutput.model_validate(data)

    # 2) JSON dict attached
    if hasattr(result, "json_dict") and result.json_dict:
        return ImageGenerationOutput.model_validate(result.json_dict)

    # 3) raw JSON string
    if hasattr(result, "raw") and result.raw:
        try:
            return ImageGenerationOutput.model_validate_json(result.raw)
        except Exception:
            pass

    # 4) direct dict or JSON string
    if isinstance(result, dict):
        return ImageGenerationOutput.model_validate(result)
    if isinstance(result, str):
        return ImageGenerationOutput.model_validate_json(result)

    # 5) last-resort: attributes on CrewOutput (some builds expose them)
    try:
        data = {
            "id": getattr(result, "id"),
            "image_url": getattr(result, "image_url"),
            "prompt_used": getattr(result, "prompt_used"),
        }
        return ImageGenerationOutput.model_validate(data)
    except Exception:
        raise RuntimeError(f"Unexpected agent output type: {type(result)} {result}")


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


def generate_images(paths: dict, scenario: str, n: int = 10) -> dict:
    """Generate n images into an existing scenario (uses provided paths)."""
    images_dir: Path = paths["images"]
    info_json: Path = paths["images_info_path"]
    root_dir: Path = paths["root"]

    for i in range(n):
        print(f"[Step 1] Generating image {i+1}/{n}…")
        crew = build_generation_crew()
        result = crew.kickoff(inputs={"prompt": scenario})
        out = _parse_generation_output(result)

        image_fname = f"{out.id}.png"
        image_path = images_dir / image_fname
        download_image(str(out.image_url), image_path)

        # store path relative to the scenario root (portable across machines)
        try:
            relpath = str(image_path.relative_to(root_dir))
        except Exception:
            import os as _os
            relpath = _os.path.relpath(str(image_path), start=str(root_dir))

        _append_json_array(
            info_json,
            {
                "id": out.id,
                "image_url": str(out.image_url),
                "prompt_used": out.prompt_used,
                "filename": image_fname,
                "relpath": relpath,
                "model": os.getenv("IMAGE_MODEL", "gpt-image-1"),
            },
        )
        print(f"✅ Saved image {out.id} -> {image_path}  URL: {out.image_url}")

    print(f"[Step 1] Done. Images + metadata saved under: {paths['root']}")
    return paths


def run_describe_images(paths: dict) -> None:
    """Reads images_info.json and produces one JSON description per image.
    Saves to {descriptions}/{image_id}.json
    Uses **LOCAL FILE PATH** for this pipeline hop.
    """
    descriptions_dir: Path = paths["descriptions"]
    info_json: Path = paths["images_info_path"]
    images_dir: Path = paths["images"]
    root_dir: Path = paths["root"]

    if not info_json.exists():
        print("No images_info.json found — nothing to describe.")
        return

    records = json.loads(info_json.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        print("images_info.json is empty — nothing to describe.")
        return

    crew = build_description_crew()

    for idx, rec in enumerate(records, start=1):
        image_id = rec.get("id")
        relpath = rec.get("relpath")
        abspath = rec.get("abspath")
        filename = rec.get("filename")

        # Resolve to a local path (prefer relpath -> abspath -> images_dir/filename)
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

        print(f"[Step 2] Describing image {idx}/{len(records)} — {image_id} (local: {image_path})")
        # Pass local file path (URL left empty)
        result = crew.kickoff(inputs={
            "image_path": str(image_path),
            "image_url": "",
        })

        desc = _parse_description_output(result)
        desc.image_id = image_id
        out_path = descriptions_dir / f"{image_id}.json"
        out_path.write_text(desc.model_dump_json(indent=2), encoding="utf-8")
        print(f"📝 Saved description -> {out_path}")

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
        print("No descriptions found — skipping bias analysis.")
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

    print(f"📊 Bias summary  -> {out_path}")
    print(f"🧭 Bias report   -> {biases_dir / 'bias_report.json'} (or bias_report_raw.txt)")

def _parse_question_set_output(result) -> QuestionSet:
    if hasattr(result, "pydantic") and result.pydantic is not None:
        p = result.pydantic
        return p if isinstance(p, QuestionSet) else QuestionSet.model_validate(
            p.model_dump() if hasattr(p, "model_dump") else p
        )
    if hasattr(result, "json_dict") and result.json_dict:
        return QuestionSet.model_validate(result.json_dict)
    if hasattr(result, "raw") and result.raw:
        return QuestionSet.model_validate_json(result.raw)
    if isinstance(result, dict):
        return QuestionSet.model_validate(result)
    if isinstance(result, str):
        return QuestionSet.model_validate_json(result)
    raise RuntimeError(f"Unexpected question-set output: {type(result)} {result}")


def run_generate_questions(paths: dict) -> list[dict]:
    """Reads biases/bias_report.json -> writes questions/questions.json using OSS-20B."""
    biases_dir: Path = paths["biases"]
    questions_dir: Path = paths["questions"]
    questions_dir.mkdir(parents=True, exist_ok=True)

    bias_report_path = biases_dir / "bias_report.json"
    if not bias_report_path.exists():
        print("No bias_report.json found — skipping question generation.")
        return []

    bias_json = json.loads(bias_report_path.read_text(encoding="utf-8"))

    crew = build_question_writer_crew()
    result = crew.kickoff(inputs={"report_json": json.dumps(bias_json, ensure_ascii=False)})
    qset = _parse_question_set_output(result)

    # Save in your project under questions/
    out_payload = {
        "questions_list": qset.questions_list,
        "reason_of_question": qset.reason_of_question,
    }
    out_path = questions_dir / "questions.json"
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"❓ Questions written -> {out_path}")

    # Return paired list for convenience (optional)
    paired = [{"question": q, "reason": r} for q, r in zip(qset.questions_list, qset.reason_of_question)]
    return paired

def run_fact_check(paths: dict, questions: list) -> list[FactResult]:
    """Runs a fact-check per question and writes research/fact_results.json.
    Accepts either List[CheckQuestion] or List[dict] with {"question","reason"}.
    """
    from pathlib import Path
    from schemas.questions import FactResult  # ensure available

    research_dir: Path = paths.get("research") or (paths["questions"] / "research")
    research_dir.mkdir(parents=True, exist_ok=True)

    if not questions:
        print("No questions to fact-check.")
        return []

    checker_crew = build_fact_checker_crew()
    results: list[FactResult] = []

    for idx, q in enumerate(questions, start=1):
        # Robustly extract the question text from either a Pydantic object or dict
        q_text = (
            getattr(q, "question", None)
            if not isinstance(q, dict)
            else q.get("question")
        )
        if not q_text:
            print(f"Skipping empty/malformed question payload: {q}")
            continue

        print(f"[Step 4] Fact-check {idx}/{len(questions)} — {q_text}")
        res = checker_crew.kickoff(inputs={"question": q_text})

        try:
            payload = res.json_dict if hasattr(res, "json_dict") and res.json_dict else json.loads(
                res.raw if hasattr(res, "raw") and res.raw else str(res)
            )
        except Exception:
            payload = []

        if isinstance(payload, list) and payload:
            item = payload[0] or {}
            answer = item.get("answer", "NOT FOUND")
            source = item.get("source", None)
            status = "FOUND" if (answer and answer != "NOT FOUND") else "NOT_FOUND"
            results.append(FactResult(question=q_text, answer=str(answer), source=source, status=status))
        else:
            results.append(FactResult(question=q_text, answer="NOT FOUND", source=None, status="NOT_FOUND"))

    # Save under research/
    facts_path = research_dir / "fact_results.json"
    facts_payload = {"results": [r.model_dump() for r in results]}
    facts_path.write_text(json.dumps(facts_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"🔎 Fact results -> {facts_path}")
    return results

def _parse_consensus_output(result) -> ConsensusOutput:
    # Prefer pydantic, then json_dict, then raw
    if hasattr(result, "pydantic") and result.pydantic is not None:
        p = result.pydantic
        try:
            return ConsensusOutput.model_validate(p.model_dump())
        except Exception:
            return ConsensusOutput.model_validate(p.dict())
    if hasattr(result, "json_dict") and result.json_dict:
        return ConsensusOutput.model_validate(result.json_dict)
    if hasattr(result, "raw") and result.raw:
        return ConsensusOutput.model_validate_json(result.raw)
    if isinstance(result, dict):
        return ConsensusOutput.model_validate(result)
    if isinstance(result, str):
        return ConsensusOutput.model_validate_json(result)
    raise RuntimeError(f"Unexpected consensus output type: {type(result)} {result}")

def run_consensus(paths: dict, scenario_text: str | None = None) -> None:
    """Step 5 — get consensus from 5 LLMs and save JSON with real model names."""
    consensus_dir: Path = paths["consensus"]
    consensus_dir.mkdir(parents=True, exist_ok=True)

    if scenario_text is None:
        manifest = json.loads(paths["manifest_path"].read_text(encoding="utf-8"))
        scenario_text = manifest.get("scenario", "")

    crew = build_consensus_crew()
    # Pass both prompt and the ordered model names so the consensus task can write model_name correctly
    result = crew.kickoff(inputs={
        "prompt": scenario_text,
        "model_names": json.dumps(replicate_models, ensure_ascii=False)
    })

    out = _parse_consensus_output(result)
    out_path = consensus_dir / "consensus_output.json"
    out_path.write_text(json.dumps(out.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"🤝 Consensus saved -> {out_path}")


if __name__ == "__main__":
    paths = setup_scenario(SCENARIO, SCENARIO_ID)
    generate_images(paths, SCENARIO, n=N_IMAGES)
    run_describe_images(paths)
    run_analyze_bias(paths)
    qs = run_generate_questions(paths)
    run_fact_check(paths, qs)
    run_consensus(paths)