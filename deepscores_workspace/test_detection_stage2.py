#!/usr/bin/env python3
"""
Stage 2 YOLOv8 Detection 테스트 스크립트
작성일: 2025-07-29
목적: 훈련된 모델로 음악 기호 검출 테스트
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

def test_detection_on_image(model_path, image_path, output_dir="test_results", conf_threshold=0.25):
    """단일 이미지에서 검출 테스트"""
    
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 모델 로드
    print(f"📦 모델 로드: {model_path}")
    model = YOLO(model_path)
    
    # 이미지 로드
    print(f"🖼️  이미지 로드: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print("❌ 이미지를 로드할 수 없습니다.")
        return
    
    # BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 검출 실행
    print(f"🔍 검출 실행 중... (신뢰도 임계값: {conf_threshold})")
    results = model(image_rgb, conf=conf_threshold)
    
    # 결과 분석
    detections = results[0]
    boxes = detections.boxes
    
    if boxes is not None:
        print(f"✅ {len(boxes)} 개의 객체 검출됨")
        
        # 클래스별 카운트
        class_counts = {}
        for box in boxes:
            class_id = int(box.cls)
            class_name = model.names[class_id]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        print("\n📊 클래스별 검출 수:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {class_name}: {count}개")
    else:
        print("❌ 검출된 객체가 없습니다.")
    
    # 시각화
    annotated_image = results[0].plot()
    
    # 결과 저장
    output_path = output_dir / f"detection_result_{Path(image_path).stem}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print(f"💾 결과 저장: {output_path}")
    
    # 상세 결과 저장 (JSON)
    detection_data = {
        'image_path': str(image_path),
        'total_detections': len(boxes) if boxes is not None else 0,
        'confidence_threshold': conf_threshold,
        'class_counts': class_counts if boxes is not None else {},
        'detections': []
    }
    
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detection_data['detections'].append({
                'class_id': int(box.cls),
                'class_name': model.names[int(box.cls)],
                'confidence': float(box.conf),
                'bbox': [x1, y1, x2, y2]
            })
    
    json_path = output_dir / f"detection_result_{Path(image_path).stem}.json"
    with open(json_path, 'w') as f:
        json.dump(detection_data, f, indent=2)
    print(f"📄 상세 결과 저장: {json_path}")
    
    return detection_data

def test_on_validation_set(model_path, data_yaml="data_stage2_20250727/deepscores_stage2.yaml", num_samples=5):
    """검증 세트에서 랜덤 샘플 테스트"""
    import random
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 데이터셋 경로 확인
    with open(data_yaml, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.SafeLoader)
    
    val_images_dir = Path(data_config['path']) / data_config['val'].replace('images/', '')
    val_images = list(val_images_dir.glob("*.png"))
    
    # 랜덤 샘플 선택
    sample_images = random.sample(val_images, min(num_samples, len(val_images)))
    
    print(f"🎲 {len(sample_images)}개 이미지에서 테스트")
    
    for img_path in sample_images:
        test_detection_on_image(model_path, img_path)

def batch_predict(model_path, image_dir, output_dir="batch_results", conf_threshold=0.25):
    """디렉토리의 모든 이미지에 대해 예측"""
    
    # 모델 로드
    model = YOLO(model_path)
    
    # 이미지 목록
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"❌ {image_dir}에 이미지가 없습니다.")
        return
    
    print(f"📁 {len(image_files)}개 이미지 처리 중...")
    
    # 배치 예측
    results = model.predict(
        source=image_dir,
        conf=conf_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        project=output_dir,
        name="batch_predict"
    )
    
    print(f"✅ 완료! 결과는 {output_dir}/batch_predict에 저장됨")

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="YOLOv8 Detection 테스트")
    parser.add_argument("--model", default="runs/stage2/yolov8s_45classes_0728_0620/weights/best.pt", 
                        help="모델 경로")
    parser.add_argument("--image", help="테스트할 이미지 경로")
    parser.add_argument("--image-dir", help="테스트할 이미지 디렉토리")
    parser.add_argument("--val-test", action="store_true", help="검증 세트에서 랜덤 테스트")
    parser.add_argument("--num-samples", type=int, default=5, help="검증 세트 샘플 수")
    parser.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    parser.add_argument("--output", default="test_results", help="출력 디렉토리")
    
    args = parser.parse_args()
    
    if args.image:
        # 단일 이미지 테스트
        test_detection_on_image(args.model, args.image, args.output, args.conf)
    elif args.image_dir:
        # 디렉토리 배치 테스트
        batch_predict(args.model, args.image_dir, args.output, args.conf)
    elif args.val_test:
        # 검증 세트 랜덤 테스트
        test_on_validation_set(args.model, num_samples=args.num_samples)
    else:
        print("❌ --image, --image-dir, 또는 --val-test 중 하나를 지정하세요.")