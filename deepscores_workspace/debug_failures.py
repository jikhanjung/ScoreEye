#!/usr/bin/env python3
"""
실패한 이미지들의 원인 상세 분석
"""

import json
import cv2
from pathlib import Path
from collections import defaultdict

def debug_failures():
    """실패 원인 상세 분석"""
    
    # 설정 로드
    with open('stage1_preprocess_config.json', 'r') as f:
        config = json.load(f)
    
    selected_classes = set(str(cls_id) for cls_id in config['selected_classes'])
    class_mapping = config['class_mapping']
    
    # JSON 파일 로드
    json_file = "/mnt/f/ds2_complete/deepscores-complete-0_test.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    images = data['images']
    categories = data['categories']
    
    # 필터링된 어노테이션 수집
    filtered_annotations = defaultdict(list)
    if isinstance(annotations, dict):
        for ann_id, ann in annotations.items():
            cat_id = ann.get('cat_id')
            if cat_id:
                if isinstance(cat_id, list):
                    cat_id = cat_id[0] if cat_id else None
                
                if cat_id and cat_id in selected_classes:
                    image_id = ann.get('img_id') or ann.get('image_id')
                    bbox = ann.get('a_bbox') or ann.get('bbox', [0, 0, 1, 1])
                    
                    if image_id and bbox:
                        filtered_annotations[image_id].append({
                            'cat_id': cat_id,
                            'bbox': bbox,
                            'area': ann.get('area', 1)
                        })
    
    print(f"✅ 필터링 완료: {len(filtered_annotations):,d}개 이미지")
    
    # 처음 5개 이미지 분석
    image_list = list(filtered_annotations.keys())[:5]
    
    for i, image_id in enumerate(image_list, 1):
        print(f"\n{'='*60}")
        print(f"🔍 이미지 {i}: ID {image_id}")
        
        # 이미지 정보 찾기
        image_info = None
        if isinstance(images, list):
            for img in images:
                if img.get('id') == image_id or str(img.get('id')) == str(image_id):
                    image_info = img
                    break
        
        if not image_info:
            print(f"❌ 실패 원인: 이미지 정보 없음")
            print(f"   이미지 ID: {image_id} (타입: {type(image_id)})")
            print(f"   Images 타입: {type(images)}")
            if isinstance(images, list) and len(images) > 0:
                print(f"   첫 번째 이미지 샘플: ID={images[0].get('id')} (타입: {type(images[0].get('id'))})")
            continue
        
        filename = image_info.get('filename') or image_info.get('file_name', f"{image_id}.png")
        image_path = Path("/mnt/f/ds2_complete/images") / filename
        
        print(f"📁 파일명: {filename}")
        print(f"📍 경로: {image_path}")
        
        if not image_path.exists():
            print(f"❌ 실패 원인: 이미지 파일 없음")
            print(f"   경로 존재: {image_path.parent.exists()}")
            if image_path.parent.exists():
                # 비슷한 이름의 파일 찾기
                similar_files = list(image_path.parent.glob(f"*{Path(filename).stem}*"))
                print(f"   비슷한 파일들: {similar_files[:3]}")
            continue
        
        # 이미지 로드 시도
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 실패 원인: 이미지 로드 실패")
            print(f"   파일 크기: {image_path.stat().st_size} bytes")
            continue
        
        height, width = image.shape[:2]
        print(f"✅ 성공: 이미지 로드 완료")
        print(f"   크기: {width}x{height}")
        print(f"   어노테이션 수: {len(filtered_annotations[image_id])}")
        
        # 어노테이션 상세 분석
        valid_annotations = 0
        for ann in filtered_annotations[image_id][:3]:  # 처음 3개만
            bbox = ann['bbox']
            x1, y1, x2, y2 = bbox
            cat_id = ann['cat_id']
            cat_name = categories.get(str(cat_id), {}).get('name', 'Unknown')
            
            # 바운딩 박스 유효성 검사
            bbox_valid = (x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height and x1 < x2 and y1 < y2)
            if bbox_valid:
                valid_annotations += 1
            
            print(f"   📦 {cat_name}: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] {'✅' if bbox_valid else '❌'}")
        
        print(f"   📊 유효한 어노테이션: {valid_annotations}/{len(filtered_annotations[image_id])}")

if __name__ == "__main__":
    debug_failures()