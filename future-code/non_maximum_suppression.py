import os
import torch

thresh_iou = 0.0

def non_maximum_suppression(P: torch.Tensor, thresh_iou: float):
    # Ensure P is a 2D tensor with shape [N, 5]
    if P.dim() == 1:
        P = P.unsqueeze(0)
    if P.size(0) == 0:
        return torch.empty((0, 5))

    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]
    scores = P[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)

    keep = []
    while order.size(0) > 0:
        idx = order[0]
        keep.append(P[idx])

        if order.size(0) == 1:
            break

        # Get coordinates of intersections
        xx1 = torch.max(x1[order[1:]], x1[idx])
        yy1 = torch.max(y1[order[1:]], y1[idx])
        xx2 = torch.min(x2[order[1:]], x2[idx])
        yy2 = torch.min(y2[order[1:]], y2[idx])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        rem_areas = areas[order[1:]]
        union = (rem_areas - inter) + areas[idx]
        iou = inter / union

        # Keep indices with IoU <= threshold
        mask = iou <= thresh_iou
        order = order[1:][mask]

    return torch.stack(keep) if keep else torch.empty((0, 5))

def sort_data(unsorted_dir):
    txt_files = [f for f in os.listdir(unsorted_dir) if f.endswith(".txt")]
    #os.makedirs(output_dir, exist_ok=True)

    for txt_file in txt_files:
        file_path = os.path.join(unsorted_dir, txt_file)
        output_path = file_path #os.path.join(output_dir, txt_file)
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

            char_boxes.append({
                'class_id': new_class,
                'bbox': (x_center, y_center, width, height),
                'score': score
                })

        char_boxes.sort(key=lambda d: (d['bbox'][1], -d['bbox'][0]))
        for box in char_boxes:
            class_id = box['class_id']
            x_center, y_center, width, height = box['bbox']
            score = box['score']
            output_line = (
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}"
            )
            output_lines.append(output_line)

        # Write to output file
        with open(output_path, "w") as f_out:
            f_out.write("\n".join(output_lines))
        print(f"Created combined file: {output_path}")

def combine_lines(folder_list, weights, output_dir):
    assert len(folder_list) > 0, "At least one folder must be provided"
    weights = [0.5, 0.25, 0.5]
    labels_dir = os.path.join(folder_list[0], "labels")
    txt_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

    os.makedirs(output_dir, exist_ok=True)

    for txt_file in txt_files:
        all_boxes = []
        for subfolder, weight in zip(folder_list, weights):
            file_path = os.path.join(subfolder, "labels", txt_file)
            try:
                with open(file_path, "r") as f:
                    lines = f.read().splitlines()
                boxes = []
                for line in lines:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        # Assuming format: class_id x_center y_center width height
                        # Convert to x1, y1, x2, y2
                        x_center, y_center, width, height = (
                            parts[1],
                            parts[2],
                            parts[3],
                            parts[4],
                        )
                        x1 = x_center - width / 2
                        y1 = y_center - height / 2
                        x2 = x_center + width / 2
                        y2 = y_center + height / 2
                        score = 1.0  # Default score if not provided
                        if len(parts) >= 6:
                            score = parts[5]
                        # Apply weight to score
                        score *= weight
                        boxes.append([x1, y1, x2, y2, score])
                if boxes:
                    all_boxes.append(torch.tensor(boxes))
                else:
                    all_boxes.append(torch.empty((0, 5)))
            except FileNotFoundError:
                print(f"File {file_path} not found. Skipping.")
                all_boxes.append(torch.empty((0, 5)))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                all_boxes.append(torch.empty((0, 5)))

        # Combine all boxes into a single tensor
        combined_boxes = (
            torch.cat(all_boxes, dim=0) if all_boxes else torch.empty((0, 5))
        )
        if combined_boxes.size(0) == 0:
            # No boxes to process
            selected_boxes = []
        else:
            # Apply NMS
            selected_boxes = non_maximum_suppression(combined_boxes, thresh_iou)

        # Convert selected boxes back to original format
        output_lines = []
        for box in selected_boxes:
            x1, y1, x2, y2, score = box.tolist()
            width = x2 - x1
            height = y2 - y1
            x_center = x1 + width / 2
            y_center = y1 + height / 2
            # Assuming class_id is 0 (modify if needed)
            output_line = (
                f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}"
            )
            output_lines.append(output_line)

        # Write to output file
        output_path = os.path.join(output_dir, txt_file)
        with open(output_path, "w") as f_out:
            f_out.write("\n".join(output_lines))
        print(f"Created combined file: {output_path}")


if __name__ == "__main__":
    folder_list = ["nom-detection-gt/yolo10m-1280-tkh-mth", "nom-detection-gt/yolo10m-1280-tkh", "nom-detection-gt/yolo11n-1280-tkh"]
    weights = [0.5, 0.25, 0.5]
    output_dir = "nom-detection-gt/merged-0.5-0.25-0.5"
    combine_lines(folder_list, weights, output_dir)
    sort_data(unsorted_dir=output_dir)