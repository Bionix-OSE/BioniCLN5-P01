from ultralytics import YOLO
from os import environ
environ["CUDA_VISIBLE_DEVICES"] = "MIG-822aef03-bf94-5d72-bd27-dd86770c43e9"

model = YOLO("runs/detect/lab-runs/week-3-mth/weights/best.pt")
model.predict(
    "datasets/nomnaocr/images/DVSKTT-1 Quyen thu",
    save=True,
    conf=0.25,
    project="lab-runs",
    name="week-3-predict",
)
