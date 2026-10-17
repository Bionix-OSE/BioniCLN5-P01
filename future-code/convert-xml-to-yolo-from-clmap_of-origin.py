import os
import re
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import base64
from pathlib import Path
import shutil
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

class NomToYOLOConverter:
    def __init__(self, data_dir="datasets/nom_data", output_dir="yolo_data"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.class_mapping = {}
        self.load_class_mapping()
        
        # Create output directory structure
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "detection"), exist_ok=True)
        
    def load_class_mapping(self):
        """Load the class mapping from class_mapping.txt"""
        mapping_file = os.path.join(self.data_dir, "class_mapping.txt")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        # The first value is the class index, the second is the Unicode
                        class_idx = int(parts[0])
                        unicode_hex = parts[1]
                        self.class_mapping[unicode_hex] = class_idx - 2  # Adjust index to start from 0
        else:
            print(f"Warning: Class mapping file not found at {mapping_file}")
    
    def decode_image_base64(self, base64_string):
        """Decode the base64 encoded image string from the XML file"""
        # Pad the base64 string if needed
        padded_base64 = base64_string + '=' * (4 - len(base64_string) % 4) if len(base64_string) % 4 else base64_string
        try:
            img_data = base64.b64decode(padded_base64)
            # Convert to numpy array
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            
            # Determine height and width based on the array size
            # This is a simplified approach - in practice, you might need to determine the actual dimensions
            # from the XML or through other means
            side_length = int(np.sqrt(len(img_array)))
            
            # Reshape into a square image (this is an assumption, adjust as needed)
            img = img_array.reshape((side_length, side_length))
            return img
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None
    
    def parse_xml_annotation(self, xml_file):
        """Parse the XML annotation file and extract bounding box info"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get image filename and dimensions
            page_elem = root.find(".//npage")
            if page_elem is None:
                print(f"No npage element found in {xml_file}")
                return None
                
            image_file = page_elem.get("image_file")
            img_width = int(page_elem.get("image_width"))
            img_height = int(page_elem.get("image_height"))
            
            annotations = []
            
            # Process each nom_rectangle element
            for rect_elem in root.findall(".//nom_rectangle"):
                char_info = rect_elem.find("char_info")
                shape_info = rect_elem.find("shape")
                
                if char_info is None or shape_info is None:
                    continue
                    
                # Get unicode value and label
                unicode_hex = char_info.get("unicode")
                char_label = char_info.get("label", "")
                char_index = int(char_info.get("char_index", "0"))
                
                # Get bounding box coordinates
                x = int(shape_info.get("x"))
                y = int(shape_info.get("y"))
                width = int(shape_info.get("width"))
                height = int(shape_info.get("height"))
                
                # Map unicode to class ID
                if unicode_hex in self.class_mapping:
                    class_id = self.class_mapping[unicode_hex]
                else:
                    # Use character index directly if not in mapping
                    class_id = char_index
                
                # Convert to YOLO format (x_center, y_center, width, height) normalized
                x_center = (x + width/2) / img_width
                y_center = (y + height/2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                annotations.append({
                    'class_id': class_id,
                    'unicode': unicode_hex,
                    'label': char_label,
                    'bbox': [x_center, y_center, norm_width, norm_height],
                    'raw_bbox': [x, y, width, height]  # Keep the original bbox for visualization
                })
            
            return {
                'image_file': image_file,
                'width': img_width,
                'height': img_height,
                'annotations': annotations
            }
        except Exception as e:
            print(f"Error parsing {xml_file}: {e}")
            return None
    
    def save_yolo_annotation(self, annotations, output_file):
        """Save annotations in YOLO format"""
        with open(output_file, 'w') as f:
            for anno in annotations:
                bbox = anno['bbox']
                class_id = anno['class_id']
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    def create_visualization(self, image_path, annotation_data, output_path):
        """Create a visualization of the image with bounding boxes using PIL (matching the requested style)"""
        try:
            # Open image with PIL
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)
            
            # Define different colors for each class (cycling through 5 colors as in the provided code)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            font_size = 20
            font = ImageFont.truetype("assets/NomNaTong-Regular.otf", size=font_size)

            # Draw bounding boxes
            for anno in annotation_data['annotations']:
                x, y, w, h = anno['raw_bbox']
                class_id = anno['class_id']
                label = anno['label'] if anno['label'] else anno['unicode']
                
                # Select color based on class_id
                color = colors[class_id % len(colors)]
                
                # Draw rectangle
                draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
                
                # Draw label text
                label_text = f"{label}"
                draw.text((x - font_size//2, y - (font_size + 2)), label_text, fill=color, font=font, language="ja")
            
            # Save the visualization
            img.save(output_path)
            return True
        except Exception as e:
            print(f"Error creating visualization: {e} {image_path} -> {output_path}")
            return False
    
    def process_all_files(self):
        """Process all XML files in the data directory"""
        xml_files = [f for f in os.listdir(self.data_dir) if f.endswith('.nmp')]
        
        print(f"Found {len(xml_files)} XML files to process")
        
        for xml_file in tqdm(xml_files, desc="Processing XML files"):
            xml_path = os.path.join(self.data_dir, xml_file)
            
            # Parse XML
            annotation_data = self.parse_xml_annotation(xml_path)
            if annotation_data is None:
                continue
                
            # Get corresponding image file
            image_filename = annotation_data['image_file']
            image_path = os.path.join(self.data_dir, image_filename)
            
            if not os.path.exists(image_path):
                print(f"Image file not found: {image_path}")
                continue
                
            # Copy image to output directory
            dest_image_path = os.path.join(self.output_dir, "images", image_filename)
            shutil.copy(image_path, dest_image_path)
            
            # Save YOLO annotation
            label_filename = os.path.splitext(image_filename)[0] + '.txt'
            label_path = os.path.join(self.output_dir, "labels", label_filename)
            self.save_yolo_annotation(annotation_data['annotations'], label_path)
            
            # Create visualization
            detection_path = os.path.join(self.output_dir, "detection", image_filename)
            self.create_visualization(image_path, annotation_data, detection_path)
            # break
            
        print(f"Processed {len(xml_files)} XML files")

    def create_class_mapping_file(self):
        """Create a class mapping file with character indices, unicode values, and labels"""
        # Collect all class info from annotations
        class_info = {}
        
        xml_files = [f for f in os.listdir(self.data_dir) if f.endswith('.nmp')]
        for xml_file in tqdm(xml_files, desc="Extracting class information"):
            xml_path = os.path.join(self.data_dir, xml_file)
            
            # Parse XML
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Get character info
                for char_info in root.findall('.//char_info'):
                    char_index = char_info.get('char_index')
                    unicode_value = char_info.get('unicode')
                    label = char_info.get('label', '')
                    
                    if char_index and unicode_value:
                        class_info[char_index] = {
                            'unicode': unicode_value,
                            'label': label
                        }
            except Exception as e:
                print(f"Error parsing {xml_file} for class mapping: {e}")
                continue
        
        # Write mapping to file
        mapping_path = os.path.join(self.output_dir, "class_mapping.txt")
        with open(mapping_path, 'w', encoding='utf-8') as f:
            for char_index, data in sorted(class_info.items(), key=lambda x: int(x[0])):
                f.write(f"{char_index}\t{data['unicode']}\t{data['label']}\n")
        
        print(f"Created class mapping with {len(class_info)} characters at {mapping_path}")
        return class_info

    def create_yolo_dataset_files(self):
        """Create train.txt, val.txt, and test.txt files for YOLO training"""
        # Get all image files
        image_files = [os.path.join(self.output_dir, "images", f) 
                      for f in os.listdir(os.path.join(self.output_dir, "images")) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle the image files
        import random
        random.shuffle(image_files)
        
        # Split into train (80%), val (10%), test (10%)
        train_split = int(0.8 * len(image_files))
        val_split = int(0.9 * len(image_files))
        
        train_files = image_files[:train_split]
        val_files = image_files[train_split:val_split]
        test_files = image_files[val_split:]
        
        # Write to files
        with open(os.path.join(self.output_dir, "train.txt"), 'w') as f:
            for file in train_files:
                f.write(f"{file}\n")
                
        with open(os.path.join(self.output_dir, "val.txt"), 'w') as f:
            for file in val_files:
                f.write(f"{file}\n")
                
        with open(os.path.join(self.output_dir, "test.txt"), 'w') as f:
            for file in test_files:
                f.write(f"{file}\n")
        
        print(f"Created dataset split files: train ({len(train_files)}), val ({len(val_files)}), test ({len(test_files)})")
    
    def create_yolo_yaml_config(self, class_info):
        """Create YOLO configuration file (data.yaml)"""
        # Get number of classes
        num_classes = len(class_info)
        
        yaml_content = f"""# YOLO data configuration
train: {os.path.join(self.output_dir, 'train.txt')}
val: {os.path.join(self.output_dir, 'val.txt')}
test: {os.path.join(self.output_dir, 'test.txt')}

# Number of classes
nc: {num_classes}

# Class names
names: ["""
        
        for i in range(num_classes):
            yaml_content += f'"{i}"'
            if i < num_classes - 1:
                yaml_content += ", "
                
        yaml_content += "]\n"
        
        with open(os.path.join(self.output_dir, "data.yaml"), 'w') as f:
            f.write(yaml_content)
        
        print(f"Created YOLO config file with {num_classes} classes")

    def verify_classes(self):
        """Print statistics about the classes in the dataset"""
        class_counts = {}
        label_dir = os.path.join(self.output_dir, "labels")
        
        for label_file in os.listdir(label_dir):
            if not label_file.endswith('.txt'):
                continue
                
            label_path = os.path.join(label_dir, label_file)
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if class_id in class_counts:
                            class_counts[class_id] += 1
                        else:
                            class_counts[class_id] = 1
        
        print(f"Found {len(class_counts)} unique classes in the dataset")
        print(f"Top 10 most common classes: {sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]}")
        
    def split_train_val_test(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """Split the dataset into train, validation, and test sets"""
        # Create directories
        os.makedirs(os.path.join(self.output_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images", "val"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images", "test"), exist_ok=True)
        
        os.makedirs(os.path.join(self.output_dir, "labels", "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels", "val"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "labels", "test"), exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(os.path.join(self.output_dir, "images")) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle files
        np.random.shuffle(image_files)
        
        # Calculate split counts
        total = len(image_files)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        
        # Split files
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count+val_count]
        test_files = image_files[train_count+val_count:]
        
        # Move files to respective directories
        self._move_files_to_split(train_files, "train")
        self._move_files_to_split(val_files, "val")
        self._move_files_to_split(test_files, "test")
        
        print(f"Split dataset: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        
    def _move_files_to_split(self, files, split_name):
        """Move files to the specified split directory"""
        for img_file in files:
            # Move image
            src_img = os.path.join(self.output_dir, "images", img_file)
            dst_img = os.path.join(self.output_dir, "images", split_name, img_file)
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)
            
            # Move corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            src_label = os.path.join(self.output_dir, "labels", label_file)
            dst_label = os.path.join(self.output_dir, "labels", split_name, label_file)
            if os.path.exists(src_label):
                shutil.move(src_label, dst_label)


def main():
    # Initialize converter
    converter = NomToYOLOConverter()
    
    # Process all files
    converter.process_all_files()
    
    # Create class mapping file
    class_info = converter.create_class_mapping_file()
    
    # Create YOLO dataset files
    converter.create_yolo_dataset_files()
    
    # Create YOLO config
    converter.create_yolo_yaml_config(class_info)
    
    # Verify classes
    converter.verify_classes()
    
    # Split dataset (optional)
    # converter.split_train_val_test()
    
    print("""
    Conversion completed successfully!
    
    Directory structure:
    - yolo_data/
      - images/      : Contains all image files
      - labels/      : Contains all YOLO format label files
      - detection/   : Contains images with visualized bounding boxes
      - data.yaml    : YOLO configuration file
      - train.txt    : List of training images
      - val.txt      : List of validation images
      - test.txt     : List of test images
      - class_mapping.txt : Mapping between class IDs and characters
    
    Next steps: Use yolo_data/data.yaml for YOLO training
    """)


if __name__ == "__main__":
    main()