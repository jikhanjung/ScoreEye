#!/usr/bin/env python3
"""
Analyze DeepScores dataset to understand stem bbox structure
"""
import json
import sys

def analyze_stems():
    # Load the dataset
    print("Loading DeepScores dataset...")
    try:
        with open('ds2_dense/deepscores_train.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return
    
    print(f"Dataset loaded. Keys: {list(data.keys())}")
    
    # Find stem-related categories
    print("\n=== CATEGORY ANALYSIS ===")
    stem_categories = {}
    if 'categories' in data:
        print(f"Total categories: {len(data['categories'])}")
        
        # Check if categories is a dict or list
        if isinstance(data['categories'], dict):
            print("Categories stored as dict")
            # Show first few keys and values for debugging
            first_keys = list(data['categories'].keys())[:3]
            for key in first_keys:
                print(f"  Sample: {key} -> {data['categories'][key]}")
                
            # Categories stored as dict with string keys
            for cat_id, cat_info in data['categories'].items():
                cat_name = cat_info['name']
                if 'stem' in cat_name.lower():
                    stem_categories[int(cat_id)] = cat_name
                    print(f"Found stem category: ID={cat_id}, Name='{cat_name}'")
        elif isinstance(data['categories'], list):
            print("Categories stored as list")
            # Categories stored as list of objects
            for cat in data['categories']:
                if isinstance(cat, dict) and 'name' in cat:
                    if 'stem' in cat['name'].lower():
                        stem_categories[cat['id']] = cat['name']
                        print(f"Found stem category: ID={cat['id']}, Name='{cat['name']}'")
                else:
                    print(f"Unexpected category format: {cat}")
        else:
            print(f"Unknown categories format: {type(data['categories'])}")
    else:
        print("No 'categories' key found in dataset")
        return
    
    if not stem_categories:
        print("No stem categories found!")
        return
    
    # Find stem annotations
    print(f"\n=== ANNOTATION ANALYSIS ===")
    stem_annotations = []
    if 'annotations' in data:
        print(f"Total annotations: {len(data['annotations'])}")
        
        # Check the type of annotations structure
        print(f"Annotations type: {type(data['annotations'])}")
        if isinstance(data['annotations'], dict):
            first_key = list(data['annotations'].keys())[0]
            print(f"First annotation key: {first_key}")
            print(f"First annotation sample: {data['annotations'][first_key]}")
        elif isinstance(data['annotations'], list):
            print(f"First annotation sample: {data['annotations'][0]}")
        
        # Handle different annotation formats
        if isinstance(data['annotations'], dict):
            # Annotations stored as dict
            for ann_id, ann_data in data['annotations'].items():
                # Check if any of the category IDs in this annotation are stem categories
                cat_ids = ann_data.get('cat_id', [])
                if isinstance(cat_ids, list):
                    for cat_id_str in cat_ids:
                        try:
                            if cat_id_str is not None:
                                cat_id_int = int(cat_id_str)
                                if cat_id_int in stem_categories:
                                    stem_annotations.append(ann_data)
                                    if len(stem_annotations) <= 10:  # Show first 10 examples
                                        cat_name = stem_categories[cat_id_int]
                                        bbox = ann_data.get('a_bbox', ann_data.get('bbox', None))
                                        print(f"Stem annotation #{len(stem_annotations)}: category='{cat_name}', bbox={bbox}")
                                    break  # Only add once per annotation
                        except (ValueError, TypeError):
                            continue
        elif isinstance(data['annotations'], list):
            # Annotations stored as list
            for ann in data['annotations']:
                if isinstance(ann, dict):
                    cat_ids = ann.get('cat_id', [])
                    if isinstance(cat_ids, list):
                        for cat_id_str in cat_ids:
                            try:
                                if cat_id_str is not None:
                                    cat_id_int = int(cat_id_str)
                                    if cat_id_int in stem_categories:
                                        stem_annotations.append(ann)
                                        if len(stem_annotations) <= 10:  # Show first 10 examples
                                            cat_name = stem_categories[cat_id_int]
                                            bbox = ann.get('a_bbox', ann.get('bbox', None))
                                            print(f"Stem annotation #{len(stem_annotations)}: category='{cat_name}', bbox={bbox}")
                                        break  # Only add once per annotation
                            except (ValueError, TypeError):
                                continue
                else:
                    print(f"Non-dict annotation: {type(ann)} -> {ann}")
                    break
    else:
        print("No 'annotations' key found in dataset")
        return
    
    print(f"\nFound {len(stem_annotations)} total stem annotations")
    
    # Analyze bbox dimensions for first 5 examples
    print(f"\n=== BBOX DIMENSION ANALYSIS ===")
    for i, ann in enumerate(stem_annotations[:5]):
        # Find which stem category this annotation belongs to
        cat_name = "unknown"
        cat_ids = ann.get('cat_id', [])
        if isinstance(cat_ids, list):
            for cat_id_str in cat_ids:
                try:
                    if cat_id_str is not None:
                        cat_id_int = int(cat_id_str)
                        if cat_id_int in stem_categories:
                            cat_name = stem_categories[cat_id_int]
                            break
                except (ValueError, TypeError):
                    continue
        
        bbox = ann.get('a_bbox', ann.get('bbox', None))
        
        if bbox and len(bbox) >= 4:
            # bbox format is typically [x, y, width, height]
            x, y, width, height = bbox[:4]
            ratio = width / height if height > 0 else float('inf')
            
            print(f"\nStem Example {i+1}:")
            print(f"  Category: {cat_name}")
            print(f"  Bbox: {bbox}")
            print(f"  X={x}, Y={y}, Width={width}, Height={height}")
            print(f"  Width:Height Ratio = {ratio:.3f}")
            
            if height > width:
                print(f"  → VERTICAL stem (height > width)")
            elif width > height:
                print(f"  → HORIZONTAL stem (width > height) - POTENTIAL ISSUE!")
            else:
                print(f"  → SQUARE stem (width == height)")
        else:
            print(f"\nStem Example {i+1}: Invalid or missing bbox: {bbox}")
    
    # Summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    valid_bboxes = []
    for ann in stem_annotations:
        bbox = ann.get('a_bbox', ann.get('bbox', None))
        if bbox and len(bbox) >= 4 and bbox[3] > 0:  # height > 0
            valid_bboxes.append(bbox)
    
    if valid_bboxes:
        ratios = [bbox[2] / bbox[3] for bbox in valid_bboxes]  # width/height
        vertical_count = sum(1 for r in ratios if r < 1.0)
        horizontal_count = sum(1 for r in ratios if r > 1.0) 
        square_count = sum(1 for r in ratios if r == 1.0)
        
        print(f"Valid stem bboxes analyzed: {len(valid_bboxes)}")
        print(f"Vertical stems (W/H < 1.0): {vertical_count} ({vertical_count/len(ratios)*100:.1f}%)")
        print(f"Horizontal stems (W/H > 1.0): {horizontal_count} ({horizontal_count/len(ratios)*100:.1f}%)")
        print(f"Square stems (W/H = 1.0): {square_count} ({square_count/len(ratios)*100:.1f}%)")
        
        avg_ratio = sum(ratios) / len(ratios)
        print(f"Average W/H ratio: {avg_ratio:.3f}")
        
        if horizontal_count > vertical_count:
            print("\n⚠️  WARNING: Most stems are HORIZONTAL - this suggests bbox format issue!")
        else:
            print("\n✓ Most stems are VERTICAL as expected for musical stems")

if __name__ == "__main__":
    analyze_stems()