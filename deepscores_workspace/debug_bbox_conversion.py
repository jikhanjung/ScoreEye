#!/usr/bin/env python3
"""Debug script to analyze bbox conversion bug for stems"""

import json
from pathlib import Path

def analyze_stem_bboxes():
    """Analyze stem bounding boxes to find conversion bug"""
    
    # Load DeepScores annotations
    train_json = Path("ds2_dense/deepscores_train.json")
    
    if not train_json.exists():
        print("DeepScores train.json not found!")
        return
    
    with open(train_json, 'r') as f:
        coco_data = json.load(f)
    
    # Get category mapping
    categories = coco_data['categories']
    stem_cat_id = None
    
    # Find stem category ID
    for cat_id, cat_info in categories.items():
        if cat_info['name'] == 'stem':
            stem_cat_id = int(cat_id)
            print(f"Found stem category: ID={stem_cat_id}, name={cat_info['name']}")
            break
    
    if stem_cat_id is None:
        print("Stem category not found!")
        return
    
    # Analyze stem annotations
    annotations = coco_data['annotations']
    stem_count = 0
    stem_examples = []
    
    for ann_id, ann in annotations.items():
        cat_ids = ann['cat_id']
        if isinstance(cat_ids, str):
            cat_ids = [cat_ids]
        
        # Check if this annotation contains stem
        for cat_id_str in cat_ids:
            if cat_id_str and int(cat_id_str) == stem_cat_id:
                stem_count += 1
                
                # Get bbox
                a_bbox = ann['a_bbox']
                x_min, y_min, x_max, y_max = a_bbox
                
                # Calculate dimensions
                width = x_max - x_min
                height = y_max - y_min
                
                # Get image info
                img_id = int(ann['img_id'])
                
                # Store example
                stem_examples.append({
                    'ann_id': ann_id,
                    'img_id': img_id,
                    'bbox': a_bbox,
                    'width': width,
                    'height': height,
                    'aspect_ratio': width / height if height > 0 else 0
                })
                
                if len(stem_examples) >= 10:  # Get 10 examples
                    break
        
        if len(stem_examples) >= 10:
            break
    
    print(f"\nTotal stem annotations found: {stem_count}")
    print("\nStem bbox examples (original DeepScores format):")
    print("=" * 80)
    
    for i, example in enumerate(stem_examples):
        print(f"\nExample {i+1}:")
        print(f"  Annotation ID: {example['ann_id']}")
        print(f"  Image ID: {example['img_id']}")
        print(f"  BBox [x_min, y_min, x_max, y_max]: {example['bbox']}")
        print(f"  Width: {example['width']:.2f} pixels")
        print(f"  Height: {example['height']:.2f} pixels")
        print(f"  Aspect ratio (w/h): {example['aspect_ratio']:.4f}")
        
        # Check if dimensions make sense
        if example['width'] < 5 and example['height'] > 20:
            print(f"  ✓ This looks like a correct vertical stem")
        elif example['width'] > example['height']:
            print(f"  ✗ WARNING: Width > Height - stem appears horizontal!")
        else:
            print(f"  ? Unusual dimensions")
    
    # Now let's simulate the conversion to YOLO format
    print("\n" + "=" * 80)
    print("Simulating YOLO conversion for first example:")
    
    if stem_examples:
        example = stem_examples[0]
        
        # Get image dimensions (we need to find this from images data)
        img_id = example['img_id']
        img_info = None
        
        for img in coco_data['images']:
            if img['id'] == img_id:
                img_info = img
                break
        
        if img_info:
            img_width = img_info['width']
            img_height = img_info['height']
            
            print(f"\nOriginal image dimensions: {img_width} x {img_height}")
            
            # Original bbox
            x_min, y_min, x_max, y_max = example['bbox']
            print(f"Original bbox: [{x_min}, {y_min}, {x_max}, {y_max}]")
            
            # Convert to COCO format [x, y, width, height]
            x = x_min
            y = y_min
            width = x_max - x_min
            height = y_max - y_min
            print(f"COCO format: [{x}, {y}, {width}, {height}]")
            
            # Convert to YOLO format
            x_center = (x + width / 2.0) / img_width
            y_center = (y + height / 2.0) / img_height
            norm_width = width / img_width
            norm_height = height / img_height
            
            print(f"\nYOLO normalized values:")
            print(f"  x_center: {x_center:.6f}")
            print(f"  y_center: {y_center:.6f}")
            print(f"  width: {norm_width:.6f}")
            print(f"  height: {norm_height:.6f}")
            
            # Convert back to pixels to verify
            print(f"\nConverting back to pixels (1024x1024 after resize):")
            width_px = norm_width * 1024
            height_px = norm_height * 1024
            print(f"  Width: {width_px:.2f} pixels")
            print(f"  Height: {height_px:.2f} pixels")
            print(f"  Aspect ratio: {width_px/height_px:.4f}")

if __name__ == "__main__":
    analyze_stem_bboxes()