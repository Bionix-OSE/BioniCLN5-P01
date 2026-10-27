import glob, os, random, shutil

random.seed(0)
images = sorted(glob.glob("datasets/lab-char-detect/images/*.jpg"))
random.shuffle(images)

n_val = max(1, int(0.2 * len(images)))
splits = {"val": images[:n_val], "train": images[n_val:]}

for split, split_images in splits.items():
    img_out = f"datasets/lab-char-detect/images/{split}"
    lbl_out = f"datasets/lab-char-detect/labels/{split}"
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for img_path in split_images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        shutil.copy(img_path, f"{img_out}/{stem}.jpg")
        shutil.copy(f"datasets/lab-char-detect/labels_single/{stem}.txt", f"{lbl_out}/{stem}.txt")

print({k: len(v) for k, v in splits.items()})
