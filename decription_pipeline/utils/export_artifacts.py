"""
Utility to extract JSON/CSV artifacts from the dataset into a lightweight bundle.

This copies only the structured outputs (JSON/CSV) from selected dataset folders
into ``92_job_description/`` while skipping raw images and manifest-like files.
"""

import shutil
import os
from pathlib import Path
from typing import Iterable

# Top-level folders under dataset/ that we want to mirror.
DATASET_FOLDERS: tuple[str, ...] = (
    "Context-free_CF",
    "Context-aware_Unrelated_CA-U",
    "Context-aware_Related_CA-R",
    "role_counting",
)

# File base names that should never be copied even if they are JSON.
SKIP_FILENAMES: set[str] = {"metadata.json", "manifest.json", "images_info.json"}

# File extensions we care about.
ALLOWED_EXTENSIONS: set[str] = {".json", ".csv"}

def _long_path(path: Path) -> str:
    path_str = os.fspath(path)
    if os.name == "nt":
        path_str = os.path.normpath(path_str)
        if not path_str.startswith("\\\\?\\"):
            path_str = "\\\\?\\" + path_str
    return path_str

def should_copy(path: Path) -> bool:
    """Return True if the file should be copied to the export bundle."""
    if not path.is_file():
        return False
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False
    if path.name in SKIP_FILENAMES:
        return False
    return True

def _unlink(path: Path) -> None:
    try:
        os.unlink(_long_path(path))
    except FileNotFoundError:
        pass

def _rmdir(path: Path) -> None:
    try:
        os.rmdir(_long_path(path))
    except FileNotFoundError:
        pass

def _clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), key=lambda p: len(str(p)), reverse=True):
        if child.is_file():
            try:
                os.unlink(_long_path(child))
            except FileNotFoundError:
                pass
        else:
            _rmdir(child)
    _rmdir(path)

def export_dataset_artifacts(
    dataset_root: Path,
    destination_root: Path,
    folders: Iterable[str] = DATASET_FOLDERS,
) -> None:
    """
    Copy JSON/CSV artifacts from ``dataset_root`` into ``destination_root``.

    Existing destination content is removed before copying.
    """
    dataset_root = dataset_root.resolve()
    destination_root = destination_root.resolve()

    if destination_root.exists():
        shutil.rmtree(destination_root, ignore_errors=True)
    destination_root.mkdir(parents=True, exist_ok=True)

    for relative_folder in folders:
        source_dir = dataset_root / relative_folder
        if not source_dir.exists():
            continue

        for file_path in source_dir.rglob("*"):
            if not should_copy(file_path):
                continue

            relative_path = file_path.relative_to(source_dir)
            target_path = destination_root / relative_folder / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            src = _long_path(file_path)
            dst = _long_path(target_path)
            try:
                shutil.copy2(src, dst)
            except FileNotFoundError:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copyfile(src, dst)
                except FileNotFoundError:
                    print(f"[warn] Failed to copy {file_path} -> {target_path}")

def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_root = project_root / "dataset"
    destination_root = project_root / "92_job_description"
    export_dataset_artifacts(dataset_root, destination_root)
    print(f"Copied JSON/CSV artifacts to {destination_root}")


if __name__ == "__main__":
    main()
