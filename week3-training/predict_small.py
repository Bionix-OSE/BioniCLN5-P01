from pathlib import Path

from ultralytics import YOLO

# 1. Load the trained YOLO model
model = YOLO("workdir/week3-trained-mth/weights/best.pt")

# 2. Define input and output folders
image_root = Path("datasets") / "nomnaocr" / "images"
label_root = Path("workdir") / "week3-nomnaocr_labels_before_clean"

# 3. Prediction settings
conf_threshold = 0.25

# 4. Create the main output folder
label_root.mkdir(parents=True, exist_ok=True)

for story_dir in image_root.iterdir():

    if not story_dir.is_dir():
        continue

    output_story_dir = label_root / story_dir.name
    output_story_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing story: {story_dir.name}")

    # 6. Find images inside this story folder
    image_paths = [
        p for p in story_dir.iterdir()
    ]

    # 7. Run inference on each image
    for image_path in image_paths:
        results = model.predict(
            source=str(image_path),
            conf=conf_threshold,
            verbose=False
        )

        result = results[0]

        # 8. Save predicted bounding boxes in YOLO TXT format
        label_path = output_story_dir / f"{image_path.stem}.txt"

        with label_path.open("w", encoding="utf-8") as f:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    x_center, y_center, width, height = (
                        box.xywhn[0].tolist()
                    )

                    f.write(
                        f"{cls_id} "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{width:.6f} "
                        f"{height:.6f}\n"
                    )

    print(f"Saved labels to: {output_story_dir}")

print("All predictions completed!")
