from ultralytics import YOLO
project = "nom-detection"
name = "yolo11n-1280-nom-data"

# Load a model
# model = YOLO("yolov10m.pt")  # load a pretrained model (recommended for training)
model = YOLO("models/yolo11n-1280-tkh.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# Train the model
model.train(data="datasets/nom-data.yaml",
            epochs=100,
            imgsz=1280,
            project=project,
            name=name,
            single_cls=True,
            mosaic=0,
            save=True,
            patience=5,
            resume=False,
            device=[0,1,2],
            batch=60,
            cache=True,
            plots=False,
            # crop_fraction=0.5,
            pretrained=True,
            deterministic=False,
            )
