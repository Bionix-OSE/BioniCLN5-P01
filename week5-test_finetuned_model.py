
from pathlib import Path
from ultralytics import YOLO


# ==========================================
# 1. CONFIGURATION
# ==========================================

# Change this to your NEW fine-tuned best.pt
CHECKPOINT = Path(
    "workdir/nomnaocr-nakagawa-finetune/weights/best.pt"
)
YAML_ROOT = Path("datasets/combined-nom_clean-nakagawa")
TEST_ROOT = Path("datasets/nom-detection-testset")
DATA_YAML = YAML_ROOT / "test_data.yaml"
TEST_IMAGES = TEST_ROOT / "images"

IMAGE_SIZE = 640
BATCH_SIZE = 8
DEVICE = 0

OUTPUT_PROJECT = "workdir/test-evaluation-after-finetune"


# ==========================================
# 2. CHECK FILES
# ==========================================

if not CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT.resolve()}"
    )

if not TEST_IMAGES.is_dir():
    raise FileNotFoundError(
        f"Test images folder not found: {TEST_IMAGES.resolve()}"
    )

if not (TEST_ROOT / "labels").is_dir():
    raise FileNotFoundError(
        f"Test labels folder not found: {(TEST_ROOT / 'labels').resolve()}"
    )

if not DATA_YAML.is_file():
    raise FileNotFoundError(
        f"Test YAML not found: {DATA_YAML.resolve()}"
    )


# ==========================================
# 3. LOAD FINE-TUNED MODEL
# ==========================================

model = YOLO(str(CHECKPOINT))

print(f"Model: {CHECKPOINT.resolve()}")
print(f"Test dataset: {TEST_ROOT.resolve()}")


# ==========================================
# 4. EVALUATE AGAINST GROUND-TRUTH LABELS
# ==========================================

metrics = model.val(
    data=str(DATA_YAML.resolve()),
    split="test",
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    plots=True,

    show_labels=False,
    show_conf=False,

    project=OUTPUT_PROJECT,
    name="metrics",
)

print("\n========== TEST METRICS ==========")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall:    {metrics.box.mr:.4f}")
print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")


# ==========================================
# 5. PREDICT AND SAVE OUTPUTS
# ==========================================

model.predict(
    source=str(TEST_IMAGES.resolve()),
    imgsz=IMAGE_SIZE,
    conf=0.25,
    iou=0.50,
    batch=BATCH_SIZE,
    device=DEVICE,
    save=True,
    save_txt=True,
    save_conf=True,
    project=OUTPUT_PROJECT,
    show_labels=False,  
    show_conf=False,
    name="predictions",
    exist_ok=False,
)

print("\nPrediction completed.")
print(f"Metrics and plots: {OUTPUT_PROJECT}/metrics")
print(f"Predictions:       {OUTPUT_PROJECT}/predictions")

old_images_output = Path("workdir/test-evaluation-after-finetune/predictions")
images_output = old_images_output / "images"
images_output.mkdir(parents=True, exist_ok=True)

for image_path in list(old_images_output.iterdir()):
    if image_path.is_file() and image_path.suffix.lower() == ".jpg":
        image_path.rename(images_output / image_path.name)