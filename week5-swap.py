from pathlib import Path
import shutil

ROOT = Path("datasets/combined-nom_clean-nakagawa")

# Move DVSKTT-2 from train to val,
# and Tale of Kieu 1871 from val to train.
moves = [
    (
        "train",
        "val",
        "Tale of Kieu 1871",
    ),
    (
        "val",
        "train",
        "DVSKTT-2 Ngoai ky toan thu",
    ),
]


def swap_stories():
    # Validate all source and destination paths first.
    operations = []

    for source_split, target_split, story in moves:
        for kind in ("images", "labels"):
            source = (
                ROOT / kind / source_split / "nomnaocr" / story
            )
            target = (
                ROOT / kind / target_split / "nomnaocr" / story
            )

            if not source.is_dir():
                raise FileNotFoundError(
                    f"Source folder not found: {source}"
                )

            if target.exists():
                raise FileExistsError(
                    f"Destination already exists: {target}"
                )

            operations.append((source, target))

    # Move only after all paths pass validation.
    for source, target in operations:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        print(f"Moved: {source} -> {target}")

    print("\nFolder swap completed.")


if __name__ == "__main__":
    swap_stories()
