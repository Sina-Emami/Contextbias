from pathlib import Path
from datetime import datetime
import json

SCENARIOS_BASE = Path("data/scenarios")


def init_scenario_root(scenario_id: str, scenario_text: str) -> dict:
    root = SCENARIOS_BASE / scenario_id
    subfolders = {
        "images": root / "images",
        "descriptions": root / "descriptions",
    }
    for path in [root, *subfolders.values()]:
        path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "id": scenario_id,
        "scenario": scenario_text,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "paths": {key: str(value) for key, value in subfolders.items()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "root": root,
        "dataset_root": root,
        "prompt_root": root,
        **subfolders,
        "manifest_path": root / "manifest.json",
        "images_info_path": subfolders["images"] / "images_info.json",
    }
