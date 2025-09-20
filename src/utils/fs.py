from pathlib import Path
from datetime import datetime
import json
import requests

SCENARIOS_BASE = Path("data/scenarios")


def init_scenario_root(scenario_id: str, scenario_text: str) -> dict:
    root = SCENARIOS_BASE / scenario_id
    subfolders = {
        "images": root / "images",
        "raw_descriptions": root / "raw_descriptions",
        "descriptions": root / "descriptions",
        "questions": root / "questions",
        "biases": root / "biases",
        "research": root / "research",
        "consensus": root / "consensus",
    }
    for p in [root, *subfolders.values()]:
        p.mkdir(parents=True, exist_ok=True)

    manifest = {
        "id": scenario_id,
        "scenario": scenario_text,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "paths": {k: str(v) for k, v in subfolders.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "root": root,
        **subfolders,
        "manifest_path": root / "manifest.json",
        "images_info_path": subfolders["images"] / "images_info.json",
    }


def download_image(url: str, filepath: Path) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    filepath.write_bytes(r.content)