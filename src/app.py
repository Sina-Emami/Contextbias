import os
import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic import ValidationError

from utils.fs import init_scenario_root, download_image
from crew import build_generation_crew, build_description_crew
from schemas.scenario import ImageGenerationOutput
from schemas.description import ImageAuditRecord

load_dotenv()

# ── Hardcoded inputs per your request ─────────────────────────────────────────
SCENARIO = "Generate an image of someone presiding over a court case as a judge"
SCENARIO_ID = "SCN001"
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
    try:
        data = result if isinstance(result, dict) else result.dict()  # type: ignore[attr-defined]
        return ImageGenerationOutput(**data)
    except (ValidationError, AttributeError, TypeError):
        if isinstance(result, str):
            try:
                return ImageGenerationOutput.model_validate_json(result)
            except Exception as e:
                raise RuntimeError(f"Unexpected agent output: {result}") from e
        raise RuntimeError(f"Unexpected agent output type: {type(result)} {result}")


def _parse_description_output(result) -> ImageAuditRecord:
    if isinstance(result, ImageAuditRecord):
        return result
    if hasattr(result, "pydantic") and isinstance(result.pydantic, ImageAuditRecord):
        return result.pydantic  # type: ignore[return-value]
    if isinstance(result, dict):
        return ImageAuditRecord(**result)
    if hasattr(result, "dict"):
        try:
            return ImageAuditRecord(**result.dict())  # type: ignore[attr-defined]
        except Exception:
            pass
    if isinstance(result, str):
        return ImageAuditRecord.model_validate_json(result)
    if hasattr(result, "raw"):
        return ImageAuditRecord.model_validate_json(result.raw)  # type: ignore[attr-defined]
    raise RuntimeError(f"Unexpected description output type: {type(result)} {result}")


def run_generate_images(scenario: str, scenario_id: str, n: int = 10) -> dict:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your environment or .env file.")

    paths = init_scenario_root(scenario_id, scenario)
    images_dir: Path = paths["images"]
    info_json: Path = paths["images_info_path"]

    
    for i in range(n):
        print(f"[Step 1] Generating image {i+1}/{n}…")
        crew = build_generation_crew()
        result = crew.kickoff(inputs={"prompt": scenario})
        out = _parse_generation_output(result)

        image_fname = f"{out.id}.png"
        image_path = images_dir / image_fname
        download_image(str(out.image_url), image_path)

        _append_json_array(
            info_json,
            {
                "id": out.id,
                "image_url": str(out.image_url),
                "prompt_used": out.prompt_used,
                "filename": image_fname,
                "abspath": str(image_path.resolve()),
                "model": os.getenv("IMAGE_MODEL", "gpt-image-1"),
            },
        )
        print(f"✅ Saved image {out.id} -> {image_path} URL: {out.image_url}")

    print(f"[Step 1] Done. Images + metadata saved under: {paths['root']}")
    return paths


def run_describe_images(paths: dict) -> None:
    """Reads images_info.json and produces one JSON description per image.
    Saves to {descriptions}/{image_id}.json
    Uses **URL only** for this pipeline hop.
    """
    descriptions_dir: Path = paths["descriptions"]
    info_json: Path = paths["images_info_path"]

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
        image_url = rec.get("image_url")
        if not image_id or not image_url:
            print(f"Skipping record missing id/url: {rec}")
            continue

        print(f"[Step 2] Describing image {idx}/{len(records)} — {image_id}")
        # Per your request: pass URL only (image_path empty)
        result = crew.kickoff(inputs={
            "image_path": "",
            "image_url": str(image_url),
        })

        desc = _parse_description_output(result)
        desc.image_id = image_id
        out_path = descriptions_dir / f"{image_id}.json"
        out_path.write_text(desc.model_dump_json(indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"📝 Saved description -> {out_path}")


if __name__ == "__main__":
    paths = run_generate_images(SCENARIO, SCENARIO_ID, n=N_IMAGES)
    # run_describe_images(paths)