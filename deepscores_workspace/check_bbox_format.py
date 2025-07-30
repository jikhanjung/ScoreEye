#!/usr/bin/env python3
"""
DeepScores 바운딩 박스 형식 확인
"""

import json
import cv2
from pathlib import Path

def check_bbox_format():
    """바운딩 박스 형식 및 좌표계 확인"""
    
    # JSON 파일 로드
    json_file = "/mnt/f/ds2_complete/deepscores-complete-0_test.json"
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    images = data['images']
    
    # 첫 번째 이미지 정보 가져오기
    if isinstance(images, list):
        image_info = images[0]
    else:
        image_info = list(images.values())[0]
    
    filename = image_info.get('filename') or image_info.get('file_name')
    image_path = Path("/mnt/f/ds2_complete/images") / filename
    
    print(f"🖼️  이미지: {filename}")
    
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ 이미지 로드 실패")
        return
    
    height, width = image.shape[:2]
    print(f"📏 이미지 크기: {width}x{height}")
    
    # 해당 이미지의 어노테이션 찾기
    image_id = image_info.get('id')
    image_annotations = []
    
    if isinstance(annotations, dict):
        for ann_id, ann in annotations.items():
            ann_image_id = ann.get('img_id') or ann.get('image_id')
            if str(ann_image_id) == str(image_id):
                image_annotations.append(ann)
                if len(image_annotations) >= 10:  # 처음 10개만
                    break
    
    print(f"📦 어노테이션 수: {len(image_annotations)}")
    
    # 바운딩 박스 형식 분석
    for i, ann in enumerate(image_annotations[:5]):
        print(f"\n--- 어노테이션 {i+1} ---")
        
        # 모든 bbox 관련 필드 확인
        bbox_fields = ['bbox', 'a_bbox', 'o_bbox']
        for field in bbox_fields:
            if field in ann:
                bbox = ann[field]
                print(f"{field}: {bbox}")
        
        # 카테고리 정보
        cat_id = ann.get('cat_id')
        if isinstance(cat_id, list):
            cat_id = cat_id[0]
        
        category_name = data['categories'].get(str(cat_id), {}).get('name', 'Unknown')
        print(f"카테고리: {category_name} (ID: {cat_id})")
        
        # a_bbox로 분석 (가장 일반적)
        if 'a_bbox' in ann:
            bbox = ann['a_bbox']
            x, y, w, h = bbox
            
            print(f"📍 좌표 분석:")
            print(f"   X: {x} ~ {x+w} (이미지 폭: {width})")
            print(f"   Y: {y} ~ {y+h} (이미지 높이: {height})")
            
            # 범위 체크
            x_valid = 0 <= x <= width and 0 <= (x+w) <= width
            y_valid = 0 <= y <= height and 0 <= (y+h) <= height
            
            print(f"   X 범위 유효: {x_valid}")
            print(f"   Y 범위 유효: {y_valid}")
            
            if not x_valid or not y_valid:
                print(f"   ❌ 범위 초과!")
                
                # 다른 해석 시도
                print(f"   🔄 다른 해석:")
                
                # 해석 1: x,y,x2,y2 형식일 가능성
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    print(f"      x1,y1,x2,y2 해석: ({x1},{y1}) ~ ({x2},{y2})")
                    x1_valid = 0 <= x1 <= width and 0 <= x2 <= width
                    y1_valid = 0 <= y1 <= height and 0 <= y2 <= height
                    print(f"      X 범위 유효: {x1_valid}")
                    print(f"      Y 범위 유효: {y1_valid}")
                    
                    if x1_valid and y1_valid:
                        print(f"      ✅ 이 해석이 맞는 것 같습니다!")
                        print(f"      크기: {abs(x2-x1)} x {abs(y2-y1)}")

if __name__ == "__main__":
    check_bbox_format()