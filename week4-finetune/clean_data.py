
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Images organized by story folders
image_root = Path("datasets") / "nomnaocr" / "images"

# Character labels predicted by YOLO
char_label_root = Path("workdir") / "nomnaocr_labels"

# Accurate text-line labels provided by the teacher
line_label_root = Path("datasets") / "nomnaocr" / "line_labels"

# Output folders
clean_label_root = Path("workdir") / "nomnaocr_labels_clean"
rejected_label_root = Path("workdir") / "rejected_labels"

# Keep a character if at least this fraction of its area
# overlaps with any text-line polygon.
overlap_threshold = 0.5


# ============================================================
# 2. READ CHARACTER LABELS (YOLO FORMAT)
# ============================================================

def read_character_labels(txt_path):
    """
    Read character labels in YOLO format.

    Each line:
    class_id x_center y_center width height [confidence]

    Coordinates are normalized to [0, 1].

    Return:
        List of (original_line, normalized_xywh)
    """
    labels = []

    with txt_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            original_line = line.strip()

            if not original_line:
                continue

            values = original_line.split()

            if len(values) < 5:
                print(
                    f"WARNING: Invalid character label: "
                    f"{txt_path}: {original_line}"
                )
                continue

            try:
                # Validate class ID and coordinates
                int(values[0])
                x, y, w, h = map(float, values[1:5])
            except ValueError:
                print(
                    f"WARNING: Invalid values: "
                    f"{txt_path}: {original_line}"
                )
                continue

            if w <= 0 or h <= 0:
                continue

            labels.append(
                (original_line, (x, y, w, h))
            )

    return labels


# ============================================================
# 3. READ TEXT-LINE LABELS (POLYGON FORMAT)
# ============================================================

def read_line_polygons(txt_path):
    """
    Read text-line annotations.

    Each line:
    x1,y1,x2,y2,x3,y3,x4,y4,text

    The first 8 values are pixel coordinates.
    The remaining text content is ignored.

    Return:
        List of polygons, each with shape (4, 2).
    """
    polygons = []

    with txt_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Split at the first 8 commas:
            # 8 coordinate values + remaining text
            parts = line.split(",", 8)

            if len(parts) < 9:
                print(
                    f"WARNING: Invalid line annotation: "
                    f"{txt_path}: {line}"
                )
                continue

            try:
                coords = [
                    float(value) for value in parts[:8]
                ]
            except ValueError:
                print(
                    f"WARNING: Invalid polygon coordinates: "
                    f"{txt_path}: {line}"
                )
                continue

            polygon = np.array(
                [
                    [coords[i], coords[i + 1]]
                    for i in range(0, 8, 2)
                ],
                dtype=np.float32
            )

            if abs(cv2.contourArea(polygon)) <= 0:
                continue

            polygons.append(polygon)

    return polygons


# ============================================================
# 4. FIND THE IMAGE MATCHING A LABEL FILE
# ============================================================

def find_matching_image(relative_label_path):
    """
    Find the image with the same relative folder and filename
    as the character TXT file.
    """

    image_path = image_root / relative_label_path.with_suffix(".jpg")

    if image_path.exists():
        return image_path

    return None


# ============================================================
# 5. CALCULATE CHARACTER-TO-LINE OVERLAP
# ============================================================

def character_overlap_ratio(
    char_xywh,
    line_polygon,
    image_width,
    image_height
):
    """
    Calculate:

        intersection_area / character_box_area

    The character box is normalized YOLO xywh.
    The text-line polygon uses absolute pixel coordinates.
    """

    x_center, y_center, width, height = char_xywh

    # Convert normalized character coordinates to pixels
    cx = x_center * image_width
    cy = y_center * image_height
    bw = width * image_width
    bh = height * image_height

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # Build the character rectangle as a polygon
    char_polygon = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ],
        dtype=np.float32
    )

    char_area = bw * bh

    if char_area <= 0:
        return 0.0

    # Calculate the actual intersection with the line polygon
    try:
        intersection_area, _ = cv2.intersectConvexConvex(
            char_polygon,
            line_polygon
        )
    except cv2.error:
        return 0.0

    return max(0.0, float(intersection_area)) / char_area


# ============================================================
# 6. FILTER ALL CHARACTER LABEL FILES
# ============================================================

def main():
    clean_label_root.mkdir(parents=True, exist_ok=True)
    rejected_label_root.mkdir(parents=True, exist_ok=True)

    total_files = 0
    total_kept = 0
    total_rejected = 0
    skipped_files = 0

    for char_file in char_label_root.rglob("*.txt"):

        # Preserve the relative story folder structure
        relative_path = char_file.relative_to(char_label_root)

        line_file = line_label_root / relative_path
        image_path = find_matching_image(relative_path)

        # Do not process a file without its reference labels
        if not line_file.is_file():
            print(f"SKIP: Missing line label: {line_file}")
            skipped_files += 1
            continue

        if image_path is None:
            print(f"SKIP: Missing image for: {char_file}")
            skipped_files += 1
            continue

        # Read the image dimensions
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"SKIP: Cannot read image: {image_path}")
            skipped_files += 1
            continue

        image_height, image_width = image.shape[:2]

        # Read both annotation formats
        char_labels = read_character_labels(char_file)
        line_polygons = read_line_polygons(line_file)

        # Avoid accidentally rejecting every character when
        # the reference file is empty or cannot be parsed.
        if not line_polygons:
            print(
                f"SKIP: No valid line polygons in {line_file}"
            )
            skipped_files += 1
            continue

        kept_lines = []
        rejected_lines = []

        for original_line, char_xywh in char_labels:

            # Find the best overlap with any text-line polygon
            best_overlap = max(
                (
                    character_overlap_ratio(
                        char_xywh,
                        polygon,
                        image_width,
                        image_height
                    )
                    for polygon in line_polygons
                ),
                default=0.0
            )

            if best_overlap >= overlap_threshold:
                kept_lines.append(original_line)
            else:
                rejected_lines.append(original_line)

        # Save cleaned character labels
        clean_file = clean_label_root / relative_path
        clean_file.parent.mkdir(parents=True, exist_ok=True)

        with clean_file.open("w", encoding="utf-8") as f:
            for label in kept_lines:
                f.write(label + "\n")

        # Save rejected labels separately for inspection
        rejected_file = rejected_label_root / relative_path
        rejected_file.parent.mkdir(parents=True, exist_ok=True)

        with rejected_file.open("w", encoding="utf-8") as f:
            for label in rejected_lines:
                f.write(label + "\n")

        total_files += 1
        total_kept += len(kept_lines)
        total_rejected += len(rejected_lines)

        print(
            f"{relative_path}: "
            f"{len(char_labels)} total, "
            f"{len(kept_lines)} kept, "
            f"{len(rejected_lines)} rejected"
        )

    print("\n========== FILTERING SUMMARY ==========")
    print(f"Processed files: {total_files}")
    print(f"Kept character boxes: {total_kept}")
    print(f"Rejected character boxes: {total_rejected}")
    print(f"Skipped files: {skipped_files}")
    print(f"Clean labels: {clean_label_root}")
    print(f"Rejected labels: {rejected_label_root}")


if __name__ == "__main__":
    main()
