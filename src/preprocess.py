# src/preprocess.py

import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

def parse_xml(xml_path):
    """Parse Pascal VOC XML to get object class and bounding box."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    classes = []
    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        classes.append(name)
        bboxes.append((xmin, ymin, xmax, ymax))
    
    return classes, bboxes

def create_segmentation_mask(image_path, xml_path, target_size=(256, 256)):
    """Create binary segmentation mask from XML annotation."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    
    mask = np.zeros(target_size, dtype=np.uint8)
    
    try:
        classes, bboxes = parse_xml(str(xml_path))
    except Exception as e:
        print(f"⚠️ Could not parse {xml_path}: {e}")
        return image, mask
    
    h, w = target_size
    
    for cls, bbox in zip(classes, bboxes):
        # Check for ANY oil-related class name
        if cls.lower() == 'oil':  # <--- This is the key fix!
            xmin, ymin, xmax, ymax = bbox
            # Scale coordinates from 640x640 to 256x256
            xmin = int(xmin * w / 640)
            xmax = int(xmax * w / 640)
            ymin = int(ymin * h / 640)
            ymax = int(ymax * h / 640)
            # Clip to valid range
            xmin = max(0, min(xmin, w-1))
            xmax = max(xmin+1, min(xmax, w))
            ymin = max(0, min(ymin, h-1))
            ymax = max(ymin+1, min(ymax, h))
            # Set mask pixels to 1
            mask[ymin:ymax, xmin:xmax] = 1
    
    return image, mask

def get_valid_pairs(data_dir):
    """Get only images that have matching XML files."""
    data_dir = Path(data_dir)
    xml_files = sorted(data_dir.glob("*.xml"))
    xml_basenames = {xml.stem for xml in xml_files}
    
    valid_pairs = []
    for jpg in data_dir.glob("*.jpg"):
        if jpg.stem in xml_basenames:
            valid_pairs.append((jpg, jpg.with_suffix('.xml')))
    
    return valid_pairs

def process_dataset(data_dir, output_dir):
    """Process all valid image-XML pairs and save as .npz files."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    valid_pairs = get_valid_pairs(data_dir)
    print(f"📊 Found {len(valid_pairs)} valid image-XML pairs")
    
    if len(valid_pairs) == 0:
        print("❌ No valid pairs found! Check your data directory.")
        return
    
    images = []
    masks = []
    oil_count = 0
    
    for i, (img_path, xml_path) in enumerate(tqdm(valid_pairs, desc="Processing")):
        try:
            image, mask = create_segmentation_mask(str(img_path), str(xml_path))
            images.append(image)
            masks.append(mask)
            if mask.sum() > 0:
                oil_count += 1
        except Exception as e:
            print(f"⚠️ Error processing {img_path}: {e}")
            continue
    
    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.uint8)
    
    print(f"\n📊 Dataset statistics:")
    print(f"   Total images: {len(images)}")
    print(f"   Images with oil: {oil_count}")
    print(f"   Oil percentage: {oil_count/len(images)*100:.1f}%")
    print(f"   Total oil pixels: {masks.sum()}")
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, masks, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Save
    np.savez_compressed(output_dir / "train.npz", images=X_train, masks=y_train)
    np.savez_compressed(output_dir / "val.npz", images=X_val, masks=y_val)
    np.savez_compressed(output_dir / "test.npz", images=X_test, masks=y_test)
    
    print(f"\n✅ Dataset saved!")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    base_dir = script_dir.parent
    
    DATA_DIR = base_dir / "data" / "raw"
    OUTPUT_DIR = base_dir / "data" / "processed"
    
    process_dataset(DATA_DIR, OUTPUT_DIR)
