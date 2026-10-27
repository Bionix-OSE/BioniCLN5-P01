
from pathlib import Path
import random
import shutil

# =========================
# 1. CONFIGURATION
# =========================

NOMNAOCR_ROOT = Path("workdir/nomnaocr_clean")

NOMNAOCR_IMAGES = NOMNAOCR_ROOT / "images"
NOMNAOCR_LABELS = NOMNAOCR_ROOT / "nomnaocr_labels_clean"
NAKAGAWA_IMAGES = Path("datasets/nakagawa-detection/images")
NAKAGAWA_LABELS = Path("datasets/nakagawa-detection/labels_single")

OUTPUT_ROOT = Path("datasets/combined-nom_clean-nakagawa")

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


# =========================
# 2. HELPER FUNCTIONS
# =========================

def copy_pair(image_path, label_path, output_image, output_label):
    """Validate and copy one image-label pair without renaming."""

    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")

    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label: {label_path}")

    # Every non-empty label line must contain 5 values.
    # Do not modify or renormalize the coordinates.
    with label_path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            values = line.split()

            if len(values) != 5:
                raise ValueError(
                    f"{label_path}, line {line_number}: "
                    f"expected 5 values, got {len(values)}"
                )

            try:
                numbers = [float(value) for value in values]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric label in {label_path}, "
                    f"line {line_number}"
                ) from exc

            if numbers[0] != 0:
                raise ValueError(
                    f"{label_path}, line {line_number}: "
                    "expected class ID 0"
                )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_label.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, output_image)
    shutil.copy2(label_path, output_label)


def split_items(items, train_ratio=TRAIN_RATIO):
    """Deterministic random split; requires at least 2 items."""
    items = list(items)

    if len(items) < 2:
        raise ValueError(
            "Need at least 2 items to create train and validation splits."
        )

    random.shuffle(items)

    train_count = round(len(items) * train_ratio)
    train_count = max(1, min(train_count, len(items) - 1))

    return items[:train_count], items[train_count:]


def copy_nomnaocr_story(story_name, split):
    """Copy one NomNaOCR story, preserving original filenames."""

    image_dir = NOMNAOCR_IMAGES / story_name
    label_dir = NOMNAOCR_LABELS / story_name

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image folder: {image_dir}")

    if not label_dir.is_dir():
        raise FileNotFoundError(f"Missing label folder: {label_dir}")

    images = sorted(image_dir.rglob("*.jpg"))

    if not images:
        raise ValueError(f"No .jpg images found in {image_dir}")

    for image_path in images:
        # Preserve any nested relative directories and filename.
        relative_path = image_path.relative_to(image_dir)
        label_path = (label_dir / relative_path).with_suffix(".txt")

        output_image = (
            OUTPUT_ROOT / "images" / split / "nomnaocr"
            / story_name / relative_path
        )

        output_label = (
            OUTPUT_ROOT / "labels" / split / "nomnaocr"
            / story_name / relative_path.with_suffix(".txt")
        )

        copy_pair(
            image_path,
            label_path,
            output_image,
            output_label
        )


def copy_nakagawa_images(images, split):
    """Copy Nakagawa images and corresponding labels."""

    for image_path in images:
        label_path = NAKAGAWA_LABELS / f"{image_path.stem}.txt"

        output_image = (
            OUTPUT_ROOT / "images" / split / "nakagawa"
            / image_path.name
        )
        output_label = (
            OUTPUT_ROOT / "labels" / split / "nakagawa"
            / f"{image_path.stem}.txt"
        )

        copy_pair(
            image_path, label_path, output_image, output_label
        )


# =========================
# 3. SPLIT NOMNAOCR BY STORY
# =========================

def main():
    random.seed(RANDOM_SEED)

    if not NOMNAOCR_ROOT.is_dir():
        raise FileNotFoundError(NOMNAOCR_ROOT)

    if not NAKAGAWA_IMAGES.is_dir():
        raise FileNotFoundError(NAKAGAWA_IMAGES)

    if not NAKAGAWA_LABELS.is_dir():
        raise FileNotFoundError(NAKAGAWA_LABELS)

    # Each immediate subdirectory represents one story.
    
    stories = sorted(
        path.name
        for path in NOMNAOCR_IMAGES.iterdir()
        if path.is_dir()
        and (NOMNAOCR_LABELS / path.name).is_dir()
    )

    if len(stories) < 2:
        raise ValueError(
            "Need at least 2 story folders with matching image and label folders."
        )

    train_stories, val_stories = split_items(stories)

    print("NomNaOCR train stories:", len(train_stories))
    print("NomNaOCR val stories:", len(val_stories))

    for story_name in train_stories:
        copy_nomnaocr_story(story_name, "train")

    for story_name in val_stories:
        copy_nomnaocr_story(story_name, "val")


    # =========================
    # 4. SPLIT NAKAGAWA BY IMAGE
    # =========================

    nakagawa_images = sorted(NAKAGAWA_IMAGES.glob("*.jpg"))

    if len(nakagawa_images) < 2:
        raise ValueError(
            "Need at least 2 Nakagawa images."
        )

    nak_train, nak_val = split_items(nakagawa_images)

    copy_nakagawa_images(nak_train, "train")
    copy_nakagawa_images(nak_val, "val")

    # =========================
    # 5. CREATE YOLO DATA YAML
    # =========================

    yaml_text = (
        f"path: {OUTPUT_ROOT.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: character\n"
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    (OUTPUT_ROOT / "data.yaml").write_text(
        yaml_text, encoding="utf-8"
    )

    print("\nDataset preparation completed.")
    print("Output:", OUTPUT_ROOT.resolve())
    print("YOLO config:", OUTPUT_ROOT / "data.yaml")


if __name__ == "__main__":
    main()
