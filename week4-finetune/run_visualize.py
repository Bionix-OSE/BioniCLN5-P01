from visualize import visualize_image_annotations
# visualize_image_annotations(
#     image_path="datasets/lab-char-detect/images/nlvnpf-0023-022.jpg",
#     txt_path="datasets/lab-char-detect/labels/nlvnpf-0023-022.txt",
#     output_path="check.jpg",
#     label_map={0: "character"},
# )

import random
from pathlib import Path

if __name__ == "__main__":
    # Input folders
    image_root = Path("datasets") / "nomnaocr" / "images"
    label_root = Path("workdir") / "nomnaocr_labels"

    # Output folder
    output_root = Path("workdir") / "visualized_samples"

    # Number of images to visualize
    num_samples = 20

    # Find images that have corresponding TXT labels
    valid_pairs = []

    for image_path in image_root.rglob("*"):

        # Preserve the story folder structure
        relative_path = image_path.relative_to(image_root)
        label_path = (label_root / relative_path).with_suffix(".txt")

        if label_path.exists():
            valid_pairs.append((image_path, label_path))

    # Check whether any valid pairs exist
    if not valid_pairs:
        raise RuntimeError(
            "No images with matching TXT labels were found."
        )

    # Randomly select up to 20 images
    random.seed(42)
    samples = random.sample(
        valid_pairs,
        min(num_samples, len(valid_pairs))
    )

    # Create output folder
    output_root.mkdir(parents=True, exist_ok=True)

    # Visualize selected images
    for index, (image_path, label_path) in enumerate(samples, start=1):
        output_path = output_root / f"{index:02d}_{image_path.stem}.jpg"

        visualize_image_annotations(
            str(image_path),
            str(label_path),
            str(output_path),
            {}
        )

        print(f"[{index}/{len(samples)}] Saved: {output_path}")

    print(f"\nDone! Visualized images are in: {output_root}")
