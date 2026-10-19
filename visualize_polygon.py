from pathlib import Path
import cv2
import numpy as np


def visualize_line_polygon_annotations(
    image_path,
    txt_path,
    output_path,
):
    """
    Visualize text-line polygon annotations.

    TXT format:
    x1,y1,x2,y2,x3,y3,x4,y4,text

    The text content is ignored.
    """

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    txt_path = Path(txt_path)
    output_path = Path(output_path)

    # Read annotations
    with txt_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # Split into 9 parts at most:
            # 8 coordinates + remaining text
            parts = line.split(",", 8)

            if len(parts) < 9:
                print(f"Skipping invalid annotation: {line}")
                continue

            try:
                coords = [float(value) for value in parts[:8]]
            except ValueError:
                print(f"Skipping invalid coordinates: {line}")
                continue

            # Convert 8 coordinates into 4 polygon vertices
            points = np.array(
                [
                    [round(coords[i]), round(coords[i + 1])]
                    for i in range(0, 8, 2)
                ],
                dtype=np.int32
            )

            # Draw polygon without displaying the text label
            cv2.polylines(
                image,
                [points],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

    # Create output directory if necessary
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(
            f"Cannot save visualization: {output_path}"
        )
