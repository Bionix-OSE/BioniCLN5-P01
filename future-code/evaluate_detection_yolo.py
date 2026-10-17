from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import DetMetrics
from ultralytics.utils.ops import xywh2xyxy


def load_labels(path):
    """Load YOLO format labels [cls, x, y, w, h]."""
    if not path.exists():
        return torch.empty((0, 5))
    data = np.loadtxt(path, ndmin=2)
    return torch.tensor(data[:, :5], dtype=torch.float32)  # Ensure we only take the first 5 columns


def evaluate(gt_dir, pred_dir, iou_thres=0.5):
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    metrics = DetMetrics()
    metrics.names = {0: "character"}

    val = DetectionValidator()

    gt_filemap = {f.stem[:f.stem.find("_jpg")]: f for f in gt_dir.glob("*.txt")}

    for file_dir in tqdm(pred_dir.glob("**/*.txt")):
        name = file_dir.stem
        gt_path = gt_filemap.get(name, None)
        if gt_path is None:
            print(f"Warning: No ground truth found for {name}, skipping.")
            continue
        pred_path = file_dir

        gt = load_labels(gt_path)  # [cls, x, y, w, h]
        pred = load_labels(pred_path)  # [cls, x, y, w, h]

        pred_boxes = xywh2xyxy(pred[:, 1:]) if len(pred) else torch.empty((0, 4))
        pred_cls = pred[:, 0].int() if len(pred) else torch.empty((0,), dtype=torch.int)
        pred_conf = torch.ones((len(pred),), dtype=torch.float32) if len(pred) else torch.empty((0,), dtype=torch.float32)

        gt_boxes = xywh2xyxy(gt[:, 1:]) if len(gt) else torch.empty((0, 4))
        gt_cls = gt[:, 0].int() if len(gt) else torch.empty((0,), dtype=torch.int)

        tp = val._process_batch({"bboxes": pred_boxes, "cls": pred_cls}, {"bboxes": gt_boxes, "cls": gt_cls})

        metrics.update_stats(
            {
                **tp,
                "conf": pred_conf,
                "target_img": np.unique(gt_cls.cpu().numpy()) if len(gt_cls) else np.array([], dtype=np.int64),
                "pred_cls": pred_cls,
                "target_cls": gt_cls,
            }
        )

    metrics.process()
    return metrics.results_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate YOLO detection results.")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth labels.")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing predicted labels.")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="IoU threshold for matching.")
    args = parser.parse_args()

    results = evaluate(args.gt_dir, args.pred_dir, args.iou_thres)
    print(results)
    # Example command:
    # python evaluate_detection_yolo.py --pred_dir nom-detection-gt/nomnaocr-val --gt_dir datasets/nomnaocr/test/labels