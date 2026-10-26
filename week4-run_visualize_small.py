from visualize import visualize_image_annotations
visualize_image_annotations(
    image_path="datasets/nomnaocr/images/DVSKTT-5 Ban ky tuc bien/DVSKTT_ban_tuc_XVI_3b.jpg",
    txt_path="nomnaocr_labels_clean/DVSKTT-5 Ban ky tuc bien/DVSKTT_ban_tuc_XVI_3b.txt",
    output_path="workdir/visualized_clean_character/DVSKTT_ban_tuc_XVI_3b.jpg",
    label_map={0: "character"},
)