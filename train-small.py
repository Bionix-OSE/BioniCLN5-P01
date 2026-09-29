from ultralytics import YOLO
from os import environ
environ["CUDA_VISIBLE_DEVICES"] = "MIG-822aef03-bf94-5d72-bd27-dd86770c43e9"

model = YOLO("yolo11n.pt") # pretrained COCO checkpoint, used as a starting point
model.train(
    data="datasets/lab-char-detect/data.yaml",
    epochs=20,
    imgsz=640,
    single_cls=True, # collapse to one class even if labels weren't pre-converted
    batch=8,
    patience=10,
    project="lab-runs",
    name="character-detect",
    device=0
)
