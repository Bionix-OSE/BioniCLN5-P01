import os


def exported(main_folder, compared_folder, output_dir):
    # Get all books in the main folder
    books = [
        f
        for f in os.listdir(main_folder)
        if os.path.isdir(os.path.join(main_folder, f))
    ]

    # Process target files from compared_folder
    target_files = [f for f in os.listdir(compared_folder) if f.endswith(".txt")]

    # Create output directory once at the start
    os.makedirs(output_dir, exist_ok=True)

    # 1. Precompute text file inventory for each book
    label_fold = {}
    for book in books:
        gts_path = os.path.join(main_folder, book, "gts")
        try:
            # Get all text files with set for fast lookups
            label_files = {
                f
                for f in os.listdir(gts_path)
                if f.endswith(".txt") and f in target_files
            }
        except FileNotFoundError:
            print(f"Missing directory: {gts_path}")
            label_files = set()
        label_fold[book] = label_files

    for filename in target_files:
        collected_lines = []

        # 3. Check all books for this file
        for book in books:
            if filename in label_fold[book]:
                file_path = os.path.join(main_folder, book, "gts", filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        collected_lines.extend(f.readlines())
                except Exception as e:
                    print(f"Error reading {file_path}: {str(e)}")

        # 4. Write results if we found any entries
        if collected_lines:
            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w", encoding="utf-8") as f:
                f.writelines(collected_lines)
            print(f"Successfully exported {filename} ({len(collected_lines)} lines)")
        else:
            print(f"No entries found for {filename}")


def sort_data(unsorted_dir, output_dir):
    txt_files = [f for f in os.listdir(unsorted_dir) if f.endswith(".txt")]
    os.makedirs(output_dir, exist_ok=True)

    for txt_file in txt_files:
        file_path = os.path.join(unsorted_dir, txt_file)
        output_path = os.path.join(output_dir, txt_file)
        try:
            with open(file_path, "r") as f:
                box_rows = f.readlines()
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

        output_lines = []
        char_boxes = []
        for row in box_rows:
            box = list(map(float, row.strip().split()))
            # Assuming format: class_id x_center y_center width height score
            class_id, x_center, y_center, width, height, score = box[:]

            new_class = int(class_id)

            char_boxes.append(
                {
                    "class_id": new_class,
                    "bbox": (x_center, y_center, width, height),
                    "score": score,
                }
            )

        char_boxes.sort(key=lambda d: (d["bbox"][1], -d["bbox"][0]))
        for box in char_boxes:
            class_id = box["class_id"]
            x_center, y_center, width, height = box["bbox"]
            score = box["score"]
            output_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}"
            output_lines.append(output_line)

        # Write to output file
        with open(output_path, "w") as f_out:
            f_out.write("\n".join(output_lines))
        print(f"Created combined file: {output_path}")


import numpy as np
from PIL import Image


def labeled_files(
    image_file, line_label_file, box_label_file, output_file, box_only=True
):
    img = np.array(Image.open(image_file))
    img_height, img_width = img.shape[:2]

    with open(line_label_file) as f:
        columns = []
        for line in f:
            parts = line.strip().split(",")
            chars = parts[8]

            try:
                coords = list(map(float, parts[:8]))
                # Convert to x1, y1, x2, y2
                x_coords = coords[::2]
                y_coords = coords[1::2]

                Xmin = min(x_coords) / img_width
                Ymin = min(y_coords) / img_height
                Xmax = max(x_coords) / img_width
                Ymax = max(y_coords) / img_height

                columns.append(
                    {
                        "bbox": (Xmin, Ymin, Xmax, Ymax),
                        "chars": chars,
                        "char_index": 0,  # Tracks next character to assign
                    }
                )

            except ValueError:
                print(f"Error parsing label: {line} of file {f}")
                continue

    with open(box_label_file) as f:
        output_lines = []
        for row in f:
            box = list(map(float, row.strip().split()))
            # Assuming format: class_id x_center y_center width height score
            class_id, x_center, y_center, width, height, score = box[:]

            new_class = int(class_id)
            for col in columns:
                Xmin, Ymin, Xmax, Ymax = col["bbox"]
                if Xmin < x_center < Xmax and Ymin < y_center < Ymax:
                    if col["char_index"] < len(col["chars"]):
                        new_class = col["chars"][col["char_index"]]
                        col["char_index"] += 1

                    if box_only:
                        output_line = (
                            f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        )
                    else:
                        output_line = f"{new_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}"
                    output_lines.append(output_line)
                    break

        # Write to output file
        with open(output_file, "w") as f_out:
            f_out.write("\n".join(output_lines))
        print(f"Created combined file: {output_file}")


if __name__ == "__main__":
    import glob
    import os

    # parse args
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter box label using text line label"
    )
    parser.add_argument(
        "--line-dataset",
        type=str,
        default="datasets/nomnaocr/Pages",
        help="Path to the line dataset",
    )
    parser.add_argument("--box-dataset", type=str, help="Path to the box dataset")
    parser.add_argument("--output-dir", type=str, help="Path to the output directory")
    args = parser.parse_args()

    dataset = args.line_dataset
    nms_data = args.box_dataset
    output_dir = args.output_dir

    img_dict = {
        os.path.basename(img)[:-4]: img
        for img in glob.glob(f"{dataset}/**/*.jpg", recursive=True)
    }
    line_labels_dict = {
        os.path.basename(txtfile)[:-4]: txtfile
        for txtfile in glob.glob(f"{dataset}/**/*.txt", recursive=True)
    }

    os.makedirs(output_dir, exist_ok=True)

    for gt in glob.glob(f"{nms_data}/**/*.txt", recursive=True):
        id = os.path.basename(gt)[:-4]
        if id not in img_dict:
            print(f"Image not found for {id}")
            continue
        if id not in line_labels_dict:
            print(f"Line label not found for {id}")
            continue
        labeled_files(img_dict[id], line_labels_dict[id], gt, f"{output_dir}/{id}.txt")

    # Example usage
    # python data_labeling.py --box-dataset nom-detection-gt/best3/labels --output-dir nom-detection-gt/best3/labeled-by-textline
