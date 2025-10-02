#!/usr/bin/env python3
from pathlib import Path
import json

# ---------- Config ----------
# If you run this next to a folder named "dataset", it'll use that.
# Otherwise it uses the current working directory as the dataset root.
DATASET_ROOT = (Path.cwd() / "dataset")
if not DATASET_ROOT.exists():
    DATASET_ROOT = Path.cwd()

MODEL_NAME = "Stable-Diffusion-XL"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}  # change if you only want png/jpg
MANIFEST_FILENAME = "manifest.json"
# ----------------------------

# Map folder names to a normalized context label
CONTEXT_MAP = {
    "CA-R": {"context-aware_related_ca-r", "context-aware_related", "ca-r"},
    "CA-U": {"context-aware_unrelated_ca-u", "context-aware_unrelated", "ca-u"},
    "CF":   {"context-free_cf", "context-free", "cf"},
}

def normalize(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

def infer_context(dir_path: Path) -> str:
    """
    Look up the closest ancestor that matches any context alias.
    Returns 'CF', 'CA-R', or 'CA-U' if found, else ''.
    """
    for ancestor in [dir_path] + list(dir_path.parents):
        name = normalize(ancestor.name)
        for label, aliases in CONTEXT_MAP.items():
            if name in aliases:
                return label
    return ""

def is_prompt_folder(p: Path) -> bool:
    """A prompt folder is any directory that directly contains image files."""
    if not p.is_dir():
        return False
    for child in p.iterdir():
        if child.is_file() and child.suffix.lower() in SUPPORTED_EXTS:
            return True
    return False

def images_in_dir(p: Path):
    """Yield image files directly in folder p (no recursion)."""
    for child in sorted(p.iterdir(), key=lambda x: x.name):
        if child.is_file() and child.suffix.lower() in SUPPORTED_EXTS:
            yield child

def main():
    root = DATASET_ROOT.resolve()
    # Walk all subfolders and collect prompt folders (those that contain images directly)
    prompt_folders = [d for d in root.rglob("*") if is_prompt_folder(d)]

    for prompt_dir in prompt_folders:
        role_folder = prompt_dir.parent.name  # one level up
        context_label = infer_context(prompt_dir)

        prompt_text = prompt_dir.name.replace("_", " ")

        items = []
        for img_path in images_in_dir(prompt_dir):
            items.append({
                "id": img_path.stem,
                "prompt_used": prompt_text,
                "filename": img_path.name,
                "relpath": img_path.relative_to(root).as_posix(),
                "model": MODEL_NAME,
                "role": role_folder,
                "context": context_label,
            })

        # write manifest.json inside the prompt folder
        manifest_path = prompt_dir / MANIFEST_FILENAME
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"Wrote {manifest_path.relative_to(root)} with {len(items)} items")

if __name__ == "__main__":
    main()
