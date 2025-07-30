#!/usr/bin/env python3
"""
어노테이션 디버깅 스크립트
"""

import json
from pathlib import Path

def debug_annotations():
    # 설정 로드
    with open('stage1_preprocess_config.json', 'r') as f:
        config = json.load(f)
    
    selected_classes = set(str(cls_id) for cls_id in config['selected_classes'])
    print(f"선택된 클래스 (샘플 10개): {list(selected_classes)[:10]}")
    
    # 첫 번째 JSON 파일 로드
    json_file = "/mnt/f/ds2_complete/deepscores-complete-0_train.json"
    print(f"\n📖 JSON 파일 로드: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    categories = data['categories']
    
    print(f"📊 전체 어노테이션 수: {len(annotations)}")
    print(f"📊 카테고리 수: {len(categories)}")
    
    # 어노테이션 샘플 확인
    print("\n🔍 어노테이션 샘플 (첫 5개):")
    sample_count = 0
    for ann_id, ann in annotations.items():
        if sample_count >= 5:
            break
        
        cat_id = ann.get('cat_id')
        image_id = ann.get('image_id')
        img_id = ann.get('img_id')  # DeepScores는 img_id를 사용할 수 있음
        bbox = ann.get('bbox')
        a_bbox = ann.get('a_bbox')  # axis-aligned bbox
        
        print(f"  {ann_id}: cat_id={cat_id}")
        print(f"    image_id={image_id}, img_id={img_id}")
        print(f"    bbox={bbox}, a_bbox={a_bbox}")
        
        # cat_id 처리
        processed_cat_id = None
        if cat_id:
            if isinstance(cat_id, list):
                processed_cat_id = cat_id[0] if cat_id else None
            else:
                processed_cat_id = cat_id
        
        print(f"    처리된 cat_id: {processed_cat_id}")
        print(f"    선택된 클래스에 포함? {processed_cat_id in selected_classes}")
        
        sample_count += 1
    
    # 실제 필터링 시뮬레이션
    print("\n🧪 필터링 시뮬레이션:")
    filtered_count = 0
    total_checked = 0
    
    for ann_id, ann in annotations.items():
        total_checked += 1
        if total_checked > 1000:  # 처음 1000개만 체크
            break
            
        cat_id = ann.get('cat_id')
        if cat_id:
            if isinstance(cat_id, list):
                cat_id = cat_id[0] if cat_id else None
            
            if cat_id and cat_id in selected_classes:
                filtered_count += 1
                if filtered_count <= 3:  # 처음 3개만 출력
                    print(f"  매칭됨: ann_id={ann_id}, cat_id={cat_id}")
    
    print(f"📊 처음 {total_checked}개 중 {filtered_count}개 매칭")
    
    # 카테고리 샘플 확인
    print("\n🔍 카테고리 샘플 (첫 10개):")
    for i, (cat_id, cat_info) in enumerate(categories.items()):
        if i >= 10:
            break
        name = cat_info.get('name', 'Unknown')
        print(f"  {cat_id}: {name}")

if __name__ == "__main__":
    debug_annotations()