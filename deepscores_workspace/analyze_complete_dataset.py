#!/usr/bin/env python3
"""
DeepScores Complete 데이터셋 분석 및 1단계 학습 준비
"""

import json
import os
from collections import Counter
from pathlib import Path
import random

def analyze_deepscores_complete():
    """전체 데이터셋 분석 및 핵심 클래스 선정"""
    
    # 첫 번째 JSON 파일로 전체 카테고리 분석
    json_path = "/mnt/f/ds2_complete/deepscores-complete-0_train.json"
    print("🔍 DeepScores Complete 데이터셋 분석 중...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 카테고리 정보
    categories = data['categories']
    print(f"\n📊 전체 클래스 수: {len(categories)}")
    
    # 어노테이션 통계
    annotations = data['annotations']
    cat_counts = Counter()
    
    # DeepScores는 annotations도 dict 형태로 저장
    if isinstance(annotations, dict):
        for ann_id, ann in annotations.items():
            cat_id = ann.get('cat_id')
            if cat_id:
                if isinstance(cat_id, list):
                    cat_id = cat_id[0]
                cat_counts[cat_id] += 1
    else:
        for ann in annotations:
            cat_id = ann.get('cat_id')
            if cat_id:
                if isinstance(cat_id, list):
                    cat_id = cat_id[0]
                cat_counts[cat_id] += 1
    
    # 카테고리 이름과 빈도 매핑
    cat_stats = []
    for cat_id, count in cat_counts.most_common():
        if str(cat_id) in categories:
            cat_name = categories[str(cat_id)]['name']
            cat_stats.append((cat_id, cat_name, count))
    
    # 상위 50개 출력
    print("\n🎯 가장 빈도가 높은 50개 클래스:")
    print("="*60)
    print(f"{'순위':>4} | {'ID':>4} | {'클래스명':<30} | {'개수':>10}")
    print("-"*60)
    
    core_classes = []
    for i, (cat_id, cat_name, count) in enumerate(cat_stats[:50], 1):
        print(f"{i:4d} | {int(cat_id):4d} | {cat_name:<30} | {count:10,d}")
        core_classes.append({
            'id': int(cat_id),
            'name': cat_name,
            'count': count,
            'rank': i
        })
    
    # 필수 추가 클래스 (빈도와 관계없이 중요한 것들)
    essential_classes = [
        'barline', 'barlineDouble', 'barlineFinal',
        'tie', 'slur', 
        'dynamicP', 'dynamicF', 'dynamicMF', 'dynamicMP',
        'clefF', 'clefC',
        'keyFlat', 'keySharp',
        'timeSig2', 'timeSig3', 'timeSig6', 'timeSig8',
        'restWhole', 'restHalf', 'restEighth', 'restSixteenth',
        'noteheadBlackBetweenLine', 'noteheadHalfBetweenLine'
    ]
    
    # 필수 클래스 중 상위 50에 없는 것 추가
    print("\n✨ 필수 추가 클래스 확인:")
    added_count = 0
    for cat_id, cat_info in categories.items():
        cat_name = cat_info['name']
        if cat_name in essential_classes:
            # 이미 core_classes에 있는지 확인
            if not any(c['name'] == cat_name for c in core_classes):
                count = cat_counts.get(int(cat_id), 0)
                if count > 100:  # 최소 100개 이상 있는 것만
                    print(f"  추가: {cat_name} (ID: {cat_id}, Count: {count:,d})")
                    core_classes.append({
                        'id': int(cat_id),
                        'name': cat_name,
                        'count': count,
                        'rank': 50 + added_count + 1
                    })
                    added_count += 1
    
    # 최종 선택된 클래스 저장
    final_classes = core_classes[:50 + added_count]
    print(f"\n📌 최종 선택된 클래스 수: {len(final_classes)}")
    
    # 클래스 매핑 저장
    class_mapping = {c['id']: idx for idx, c in enumerate(final_classes)}
    
    # 결과 저장
    result = {
        'total_classes': len(categories),
        'selected_classes': len(final_classes),
        'classes': final_classes,
        'class_mapping': class_mapping,
        'dataset_info': {
            'total_images': len(data['images']),
            'total_annotations': len(annotations),
            'annotation_distribution': dict(cat_counts.most_common(20))
        }
    }
    
    with open('stage1_classes.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("\n💾 분석 결과 저장: stage1_classes.json")
    
    # 메모리 효율적인 학습 전략 제안
    print("\n🎮 11GB GPU 학습 전략:")
    print("  - 이미지 크기: 1536x1536 (2048 대신)")
    print("  - 배치 크기: 2-4")
    print("  - 모델: YOLOv8s (medium은 메모리 초과)")
    print("  - Mixed Precision: 필수")
    print("  - Gradient Accumulation: 4 steps")
    print("  - 샘플링: 전체의 10% (약 25,000 이미지)")
    
    return final_classes

def create_stage1_config(selected_classes):
    """1단계 학습을 위한 설정 파일 생성"""
    
    # YAML 설정 생성
    yaml_config = {
        'path': '/home/jikhanjung/projects/ScoreEye/deepscores_workspace/data_stage1',
        'train': 'images/train',
        'val': 'images/val',
        'names': {idx: c['name'] for idx, c in enumerate(selected_classes)}
    }
    
    import yaml
    with open('deepscores_stage1.yaml', 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
    
    print("\n📄 Stage 1 데이터셋 설정 파일 생성: deepscores_stage1.yaml")
    
    # 전처리 스크립트 설정 생성
    preprocess_config = {
        'source_path': '/mnt/f/ds2_complete',
        'target_path': './data_stage1',
        'selected_classes': [c['id'] for c in selected_classes],
        'class_mapping': {c['id']: idx for idx, c in enumerate(selected_classes)},
        'image_size': 1536,  # 메모리 절약
        'sample_ratio': 0.1,  # 10% 샘플링
        'max_images': 25000,  # 최대 이미지 수
        'batch_size': 100,  # 전처리 배치 크기
        'num_workers': 4
    }
    
    with open('stage1_preprocess_config.json', 'w') as f:
        json.dump(preprocess_config, f, indent=2)
    
    print("⚙️ 전처리 설정 파일 생성: stage1_preprocess_config.json")
    
    return yaml_config, preprocess_config

if __name__ == "__main__":
    # 1. 데이터셋 분석 및 클래스 선정
    selected_classes = analyze_deepscores_complete()
    
    # 2. 학습 설정 파일 생성
    yaml_config, preprocess_config = create_stage1_config(selected_classes)
    
    print("\n✅ Stage 1 준비 완료!")
    print("\n다음 단계:")
    print("1. python3 preprocess_stage1.py  # 데이터 전처리")
    print("2. python3 train_stage1.py      # 학습 시작")