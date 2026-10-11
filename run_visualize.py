from visualize import visualize_image_annotations
visualize_image_annotations(
    image_path="datasets/lab-char-detect/images/nlvnpf-0023-022.jpg",
    txt_path="datasets/lab-char-detect/labels/nlvnpf-0023-022.txt",
    output_path="check.jpg",
    label_map={0: "character"},
)