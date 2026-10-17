import numpy as np
from PIL import Image

def visualize_image_annotations(image_path, txt_path, output_path, label_map):
    """
    Visualizes YOLO annotations (bounding boxes and class labels) on an image.

    This function reads an image and its corresponding annotation file in YOLO format, then
    draws bounding boxes around detected objects and labels them with their respective class names.
    The bounding box colors are assigned based on the class ID, and the text color is dynamically
    adjusted for readability, depending on the background color's luminance.

    Args:
        image_path (str): The path to the image file to annotate, and it can be in formats supported by PIL (e.g., .jpg, .png).
        txt_path (str): The path to the annotation file in YOLO format, that should contain one line per object with:
                        - class_id (int): The class index.
                        - x_center (float): The X center of the bounding box (relative to image width).
                        - y_center (float): The Y center of the bounding box (relative to image height).
                        - width (float): The width of the bounding box (relative to image width).
                        - height (float): The height of the bounding box (relative to image height).
        label_map (dict): A dictionary that maps class IDs (integers) to class labels (strings).

    Example:
        >>> label_map = {0: "cat", 1: "dog", 2: "bird"}  # It should include all annotated classes details
        >>> visualize_image_annotations("path/to/image.jpg", "path/to/annotations.txt", label_map)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from ultralytics.utils.plotting import colors

    img = np.array(Image.open(image_path))
    img_height, img_width = img.shape[:2]
    annotations = []
    with open(txt_path) as folder:
        for line in folder:
            class_id, x_center, y_center, width, height = map(float, line.split()[:5])
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            annotations.append((x, y, w, h, int(class_id)))
    fig, ax = plt.subplots(1)  # Plot the image and annotations
    for x, y, w, h, label in annotations:
        color = tuple(c / 255 for c in colors(label, True))  # Get and normalize the RGB color
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor="none")  # Create a rectangle
        ax.add_patch(rect)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]  # Formula for luminance
        # ax.text(x, y - 5, label_map[label], color="white" if luminance < 0.5 else "black", backgroundcolor=color)
    ax.imshow(img)
    # save to file
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    import glob
    import os
    input_dir = "datasets/nomnaocr/val"
    label_dir = "nom-detection-0.5-0.25-0.5"
    output_dir = label_dir + "-visualized"
    os.makedirs(output_dir, exist_ok=True)
    for img in glob.glob("datasets/nomnaocr/val/*.jpg"):
        visualize_image_annotations(img, img.replace(input_dir, label_dir).replace(".jpg", ".txt"), img.replace(input_dir, output_dir), {0: "nom"})