from ultralytics import YOLO
from os import environ
environ["CUDA_VISIBLE_DEVICES"] = "MIG-822aef03-bf94-5d72-bd27-dd86770c43e9"

model = YOLO("runs/detect/lab-runs/character-detect/weights/best.pt")
model.predict(
    "datasets/lab-char-detect/images/train",
    save=True,
    conf=0.25,
    project="lab-runs",
    name="predict",
)
