import glob, os
from prepare import convert_yolo_single_class

label_dir = "datasets/lab-char-detect/labels"
output_dir = "datasets/lab-char-detect/labels_single"
os.makedirs(output_dir, exist_ok=True)

for label_path in glob.glob(f"{label_dir}/*.txt"):
    output_path = label_path.replace(label_dir, output_dir)
    convert_yolo_single_class(label_path, output_path)
