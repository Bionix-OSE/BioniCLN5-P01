from pathlib import Path
from ultralytics import YOLO
from os import environ

environ["CUDA_VISIBLE_DEVICES"] = "MIG-822aef03-bf94-5d72-bd27-dd86770c43e9"
# ==========================================
# 1. CONFIGURATION
# ==========================================

# Change this to the path of your existing best.pt
CHECKPOINT = Path(
    "workdir/week-3-mth/weights/best.pt"
)

# Dataset created by prepare_combined_dataset.py
DATA_YAML = Path(
    "datasets/combined-nom_clean-nakagawa/data.yaml"
)

# Training settings
EPOCHS = 30
IMAGE_SIZE = 640
BATCH_SIZE = 8
PATIENCE = 10
WORKERS = 4

DEVICE = 0

PROJECT = "workdir"
RUN_NAME = "nomnaocr-nakagawa-finetune"


# ==========================================
# 2. CHECK INPUT FILES
# ==========================================

if not CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT.resolve()}"
    )

if not DATA_YAML.is_file():
    raise FileNotFoundError(
        f"Dataset YAML not found: {DATA_YAML.resolve()}"
    )


# ==========================================
# 3. LOAD PREVIOUSLY TRAINED MODEL
# ==========================================

model = YOLO(str(CHECKPOINT))

print(f"Starting checkpoint: {CHECKPOINT.resolve()}")
print(f"Dataset config: {DATA_YAML.resolve()}")


# ==========================================
# 4. FINE-TUNE ON COMBINED DATASET
# ==========================================

results = model.train(
    data=str(DATA_YAML),

    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    patience=PATIENCE,

    # One class: character
    single_cls=True,

    # Reproducibility
    seed=42,
    deterministic=True,

    # Training environment
    device=DEVICE,
    workers=WORKERS,

    # Save training results
    project=PROJECT,
    name=RUN_NAME,
    exist_ok=False,
    save=True,
    plots=True,
)

print("\nFine-tuning completed.")

print(f"Results directory: {results.save_dir}")
print(f"Best checkpoint: {results.save_dir / 'weights' / 'best.pt'}")
print(f"Last checkpoint: {results.save_dir / 'weights' / 'last.pt'}")