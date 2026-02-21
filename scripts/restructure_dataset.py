import argparse
import shutil
from pathlib import Path


def iter_role_dirs(label_dir: Path) -> list[Path]:
    """Return all role directories under a label directory, sorted for stability."""
    return sorted(
        (child for child in label_dir.iterdir() if child.is_dir()),
        key=lambda path: path.name.lower(),
    )


def pick_sentence_dir(role_dir: Path, *, longest: bool) -> Path | None:
    """Pick the sentence directory with the longest or shortest name."""
    candidates = [child for child in role_dir.iterdir() if child.is_dir()]
    if not candidates:
        return None

    key = lambda path: (len(path.name), path.name)
    return (max if longest else min)(candidates, key=key)


def remove_longest_sentence_dirs(label_dir: Path, *, dry_run: bool) -> None:
    """Delete the longest-named sentence directory inside each role directory."""
    for role_dir in iter_role_dirs(label_dir):
        target = pick_sentence_dir(role_dir, longest=True)
        if target is None:
            print(f"[skip] No sentence folders found under '{role_dir}'.")
            continue

        if dry_run:
            print(f"[dry-run] Would delete '{target}'.")
        else:
            shutil.rmtree(target)
            print(f"[delete] Removed '{target}'.")


def move_smallest_sentences_to_context_free(
    related_dir: Path, context_free_dir: Path, *, dry_run: bool
) -> None:
    """Move the shortest-named sentence directory for each role into Context-free_CF."""
    for role_dir in iter_role_dirs(related_dir):
        target = pick_sentence_dir(role_dir, longest=False)
        if target is None:
            print(f"[skip] No sentence folders left to move from '{role_dir}'.")
            continue

        dest_role_dir = context_free_dir / role_dir.name
        dest_sentence_dir = dest_role_dir / target.name

        if dest_sentence_dir.exists():
            raise RuntimeError(
                f"Destination '{dest_sentence_dir}' already exists; aborting to avoid overwriting."
            )

        if dry_run:
            print(f"[dry-run] Would ensure '{dest_role_dir}' exists.")
            print(f"[dry-run] Would move '{target}' -> '{dest_role_dir}'.")
            continue

        dest_role_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(dest_role_dir))
        print(f"[move] Moved '{dest_sentence_dir}' into Context-free_CF.")


def rename_label_directory(
    dataset_root: Path, *, current_name: str, new_name: str, dry_run: bool
) -> None:
    """Rename a top-level label directory."""
    src = dataset_root / current_name
    dst = dataset_root / new_name

    if not src.exists():
        raise FileNotFoundError(
            f"Expected '{src}' to exist before renaming to '{new_name}'."
        )
    if dst.exists():
        raise FileExistsError(
            f"Cannot rename '{current_name}' to '{new_name}' because '{dst}' already exists."
        )

    if dry_run:
        print(f"[dry-run] Would rename '{src}' -> '{dst}'.")
    else:
        src.rename(dst)
        print(f"[rename] '{current_name}' -> '{new_name}'.")


def restructure_dataset(dataset_root: Path, *, dry_run: bool) -> None:
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root '{dataset_root}' does not exist.")

    related_dir = dataset_root / "related"
    unrelated_dir = dataset_root / "unrelated"

    if not related_dir.exists() or not unrelated_dir.exists():
        raise FileNotFoundError(
            "Expected 'related' and 'unrelated' directories inside the dataset root."
        )

    context_free_dir = dataset_root / "Context-free_CF"
    if context_free_dir.exists():
        if any(context_free_dir.iterdir()):
            raise RuntimeError(
                f"'{context_free_dir}' already exists and is not empty; refusing to continue."
            )
        print(f"[info] Using existing empty directory '{context_free_dir}'.")
    elif dry_run:
        print(f"[dry-run] Would create '{context_free_dir}'.")
    else:
        context_free_dir.mkdir()
        print(f"[create] Created '{context_free_dir}'.")

    # Deletion step disabled: keep longest sentence directories.
    # remove_longest_sentence_dirs(related_dir, dry_run=dry_run)
    # remove_longest_sentence_dirs(unrelated_dir, dry_run=dry_run)

    move_smallest_sentences_to_context_free(
        related_dir, context_free_dir, dry_run=dry_run
    )

    rename_label_directory(
        dataset_root,
        current_name="related",
        new_name="Context-aware_Related_CA-R",
        dry_run=dry_run,
    )
    rename_label_directory(
        dataset_root,
        current_name="unrelated",
        new_name="Context-aware_Unrelated_CA-U",
        dry_run=dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restructure the dataset by pruning longest sentence folders, moving the "
            "shortest sentences into Context-free_CF, and renaming label directories."
        )
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        type=Path,
        default=Path.cwd() / "dataset",
        help="Path to the dataset directory (defaults to ./dataset).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the planned changes without modifying the filesystem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    restructure_dataset(args.dataset_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
