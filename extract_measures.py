#!/usr/bin/env python3
"""
Extract individual measure images from sheet music PDFs
This module uses the MeasureDetector to extract each measure as a separate image
and saves metadata for later processing.
"""

import cv2
import numpy as np
import os
import argparse
import json
import fitz  # PyMuPDF
from pathlib import Path
from datetime import datetime
from detect_measure import MeasureDetector


def extract_measures_from_page(detector, page_image, page_num, output_dir, debug=False):
    """
    Extract individual measures from a page and save them with metadata
    
    Args:
        detector: MeasureDetector instance
        page_image: Page image as numpy array
        page_num: Page number (1-indexed)
        output_dir: Base output directory
        debug: Whether to enable debug mode
        
    Returns:
        metadata: Dictionary containing measure information
    """
    # Create page directory
    page_dir = output_dir / f"page_{page_num:02d}"
    page_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect staff lines and barlines
    height, width = page_image.shape[:2]
    
    # Preprocess image
    preprocessed = detector.preprocess_image(page_image)
    
    # CRITICAL: Use consensus validation, NOT simple filtering!
    # GUI uses simple filter_barlines, but we want TRUE consensus validation
    
    # Store binary image for later use
    detector.binary_img = preprocessed
    
    # Detect staff lines first
    staff_lines_with_ranges = detector.detect_staff_lines(preprocessed)
    staff_lines = [line['y'] if isinstance(line, dict) else line for line in staff_lines_with_ranges]
    detector.staff_lines = staff_lines  # Store for later use
    
    print(f"  Detected {len(staff_lines)} staff lines")
    
    # IMPORTANT: Use detect_barlines_per_system for TRUE consensus validation
    # This method calls validate_barlines_with_consensus() internally
    validated_barlines = detector.detect_barlines_per_system(preprocessed)
    print(f"  CONSENSUS VALIDATED barlines: {len(validated_barlines)} positions: {validated_barlines}")
    
    # Get staff systems and system groups (now available after detect_barlines_per_system)
    staff_systems = getattr(detector, 'staff_systems', [])
    system_groups = detector.detect_system_groups() if staff_systems else []
    barlines_with_systems = getattr(detector, 'barlines_with_systems', [])
    
    print(f"  Detected {len(staff_systems)} staff systems")
    print(f"  Detected {len(system_groups)} system groups")
    print(f"  Barlines with system info: {len(barlines_with_systems)}")
    
    if not staff_lines:
        print(f"  No staff lines detected on page {page_num}")
        return None
        
    if not staff_systems:
        print(f"  No staff systems detected on page {page_num}")
        return None
    
    if not barlines_with_systems:
        print(f"  No system-specific barlines detected on page {page_num}")
        return None
    
    # Group barlines by SYSTEM GROUP - barlines_with_systems uses system GROUP index
    barlines_by_system_group = {}
    for bl_info in barlines_with_systems:
        system_group_idx = bl_info.get('system_idx', 0)  # This is actually system GROUP index
        x = bl_info.get('x', 0)
        
        if system_group_idx not in barlines_by_system_group:
            barlines_by_system_group[system_group_idx] = []
        barlines_by_system_group[system_group_idx].append(x)
    
    print(f"  Barlines by SYSTEM GROUP: {barlines_by_system_group}")
    
    # Initialize metadata
    metadata = {
        "page_number": page_num,
        "page_dimensions": {"width": width, "height": height},
        "staff_groups": [],
        "system_clusters": system_groups,
        "measures": [],
        "extracted_at": datetime.now().isoformat(),
        "barlines_used": barlines_by_system_group,
        "extraction_method": "CLI_consensus_validation"
    }
    
    # Store staff group information
    for i, system in enumerate(staff_systems):
        group_info = {
            "group_index": int(i),
            "staff_lines": [{"y": int(y), "index": int(j)} for j, y in enumerate(system.get('lines', []))],
            "y_range": {
                "min": int(system.get('top', 0)),
                "max": int(system.get('bottom', 0))
            },
            "center_y": int(system.get('center_y', 0)),
            "height": int(system.get('height', 0))
        }
        metadata["staff_groups"].append(group_info)
    
    # Get bracket information for measure start positions
    brackets = []
    if hasattr(detector, 'detected_brackets') and detector.detected_brackets:
        brackets = detector.detected_brackets
        print(f"  Found {len(brackets)} brackets for measure start positions")
    
    # Extract measures using SYSTEM GROUP approach (same as GUI)
    extracted_count = 0
    
    # Process each system group and apply its barlines to all systems in the group
    for group_idx, system_indices in enumerate(system_groups):
        group_barlines = barlines_by_system_group.get(group_idx, [])
        if not group_barlines:
            print(f"  System Group {group_idx}: No barlines, skipping systems {system_indices}")
            continue
            
        group_barlines_sorted = sorted(group_barlines)
        
        # Find bracket X coordinate for this system group as measure start
        bracket_x = 0  # Default fallback
        for bracket in brackets:
            bracket_systems = bracket.get('covered_staff_system_indices', [])
            # Check if this bracket covers systems in current group
            if any(sys_idx in bracket_systems for sys_idx in system_indices):
                bracket_x = bracket.get('x', 0)
                print(f"  System Group {group_idx}: Using bracket at x={bracket_x} as measure start")
                break
        
        extended_group_barlines = [bracket_x] + group_barlines_sorted
        
        print(f"  System Group {group_idx}: barlines {group_barlines_sorted}")
        print(f"    Applying to systems: {system_indices}")
        
        # Apply these barlines to ALL systems in this group
        for sys_idx in system_indices:
            if sys_idx >= len(staff_systems):
                continue
            
            system = staff_systems[sys_idx]
            print(f"  System {sys_idx} (Group {group_idx}): y={system['top']} to {system['bottom']}")
            
            # Create measures for this system using group's barlines
            system_measure_count = 0
            for i in range(len(extended_group_barlines) - 1):
                x1 = int(extended_group_barlines[i])
                x2 = int(extended_group_barlines[i + 1])
                
                # Skip if measure is too narrow
                if x2 - x1 < 20:
                    continue
                
                system_measure_count += 1
                
                # Calculate optimal Y range considering adjacent systems
                y1, y2 = detector.calculate_optimal_measure_y_range(
                    system, staff_systems, height
                )
                
                # Extract measure image
                measure_img = page_image[y1:y2, x1:x2]
                
                # Generate measure ID: P{page}_{system}_{measure}
                measure_id = f"P{page_num}_{sys_idx:02d}_{system_measure_count:03d}"
                
                # Save measure image
                measure_filename = f"{measure_id}.png"
                measure_path = page_dir / measure_filename
                cv2.imwrite(str(measure_path), measure_img)
                
                # Calculate staff lines relative to measure
                staff_lines_in_measure = []
                for j, staff_y in enumerate(system.get('lines', [])):
                    relative_y = int(staff_y - y1)
                    if 0 <= relative_y < (y2 - y1):
                        staff_lines_in_measure.append({
                            "y": int(relative_y),
                            "original_y": int(staff_y),
                            "staff_index": int(j),
                            "group_index": int(sys_idx)
                        })
                
                # Store measure metadata (same format as GUI)
                measure_info = {
                    "measure_id": str(measure_id),
                    "filename": str(measure_filename),
                    "measure_number": int(system_measure_count),
                    "staff_system_index": int(sys_idx),
                    "system_group_index": int(group_idx),
                    "bounding_box_on_page": {
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1)
                    },
                    "staff_line_coordinates_in_measure": staff_lines_in_measure
                }
                metadata["measures"].append(measure_info)
                extracted_count += 1
            
            print(f"  System {sys_idx} total measures: {system_measure_count}")
    
    # Save metadata
    metadata_path = page_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  Extracted {extracted_count} measures from page {page_num}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Extract measures from sheet music PDFs')
    parser.add_argument('input', help='Input PDF file path')
    parser.add_argument('--output', '-o', default='output/measures', 
                       help='Output directory (default: output/measures)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for PDF conversion (default: 300)')
    parser.add_argument('--pages', '-p', type=str,
                       help='Page range (e.g., "1-3" or "1,3,5")')
    parser.add_argument('--debug', '-d', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = MeasureDetector()
    
    # Process PDF
    print(f"Processing: {args.input}")
    
    # Open PDF
    pdf_document = fitz.open(args.input)
    
    # Parse page range
    if args.pages:
        if '-' in args.pages:
            start, end = map(int, args.pages.split('-'))
            page_nums = list(range(start, end + 1))
        else:
            page_nums = [int(p) for p in args.pages.split(',')]
    else:
        page_nums = list(range(1, pdf_document.page_count + 1))
    
    # Convert pages to images
    images = []
    for page_num in page_nums:
        if page_num > pdf_document.page_count:
            print(f"Warning: Page {page_num} does not exist in PDF")
            continue
        
        # Get page (0-indexed in PyMuPDF)
        page = pdf_document[page_num - 1]
        
        # Convert to image at specified DPI
        mat = fitz.Matrix(args.dpi / 72.0, args.dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to numpy array
        img_data = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        images.append(img_data)
    
    # Process each page
    all_metadata = {
        "source_file": os.path.basename(args.input),
        "dpi": args.dpi,
        "total_pages": len(images),
        "processed_pages": page_nums,
        "pages": {}
    }
    
    for page_num, img_array in zip(page_nums, images):
        print(f"\nProcessing page {page_num}...")
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            page_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            page_image = img_array
        
        # Extract measures
        page_metadata = extract_measures_from_page(
            detector, page_image, page_num, output_dir, debug=args.debug
        )
        
        if page_metadata:
            all_metadata["pages"][str(page_num)] = page_metadata
    
    # Close PDF
    pdf_document.close()
    
    # Save overall metadata
    overall_metadata_path = output_dir / "metadata.json"
    with open(overall_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtraction complete. Output saved to: {output_dir}")
    print(f"Total measures extracted: {sum(len(p['measures']) for p in all_metadata['pages'].values())}")


if __name__ == "__main__":
    main()
