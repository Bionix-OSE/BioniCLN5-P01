import numpy as np
from PIL import Image

def visualize_image_annotations(image_path, txt_path, output_path, column_path):
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
    import matplotlib.font_manager as fm
    from matplotlib.patches import Rectangle

    from ultralytics.utils.plotting import colors

    img = np.array(Image.open(image_path))
    img_height, img_width = img.shape[:2]
    annotations = []
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # Adjust path if different
    noto_font = fm.FontProperties(fname=font_path)
    with open(txt_path) as folder:
        for line in folder:
            class_id = line.split()[0]
            x_center, y_center, width, height = map(float, line.split()[1:5])
            x = (x_center - width / 2) * img_width
            y = (y_center - height / 2) * img_height
            w = width * img_width
            h = height * img_height
            annotations.append((x, y, w, h, class_id))
    fig, ax = plt.subplots(1)  # Plot the image and annotations
    for x, y, w, h, label in annotations:
        color = tuple(c / 255 for c in colors(0, True))  # Get and normalize the RGB color
        rect = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor="none")  # Create a rectangle
        ax.add_patch(rect)
        luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]  # Formula for luminance
        ax.text(x, y, label, color="black" if luminance < 0.5 else "black", bbox = None, fontsize=8, fontproperties=noto_font)  # Add text with appropriate color
    
    # columns = []
    # with open(column_path) as folder:
    #     for line in folder:
    #         parts = line.strip().split(',')
    #         class_id = parts[8]

    #         coords = list(map(float, parts[:8]))
    #         # Convert to x1, y1, x2, y2
    #         x_coords = coords[::2]
    #         y_coords = coords[1::2]
            
    #         xmin = min(x_coords)
    #         ymin = min(y_coords)
    #         xmax = max(x_coords)
    #         ymax = max(y_coords)

    #         w = xmax - xmin
    #         h = ymax - ymin

    #         columns.append((xmin, ymin, w, h, class_id))
    
    # for x, y, w, h, class_id in columns:
    #     color = tuple(c / 255 for c in	(4, 42, 255))  # Get and normalize the RGB color
    #     rect_col = Rectangle((x, y), w, h, linewidth=1, edgecolor=color, facecolor="none")  # Create a rectangle
    #     ax.add_patch(rect_col)
    #     luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]  # Formula for luminance
    #     ax.text(x, y - 5, label, color="white" if luminance < 0.5 else "black", backgroundcolor=(0, 0, 0, 0.01))

    ax.imshow(img)
    # save to file
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    import glob
    import os
    img_dir = "datasets/nomnaocr/val"
    input_dir = "nom-detection-0.5-0.25-0.5-labeled"
    output_dir = input_dir + "-visualized"
    label_dir = "nom-detection-labels"

    os.makedirs(output_dir, exist_ok=True)
    for img in glob.glob("datasets/nomnaocr/val/*.jpg"):
        visualize_image_annotations(img, img.replace(img_dir, input_dir).replace(".jpg", ".txt"), img.replace(img_dir, output_dir), img.replace(img_dir, label_dir).replace(".jpg", ".txt"))