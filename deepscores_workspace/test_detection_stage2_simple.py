#!/usr/bin/env python3
"""
Stage 2 YOLOv8 Detection 간단 테스트 스크립트
작성일: 2025-07-29
목적: 검출 결과를 원본 이미지와 txt 파일로 같은 디렉토리에 저장
"""

from ultralytics import YOLO
from pathlib import Path
import shutil
import random

def test_detection_simple(model_path, num_samples=5, conf_threshold=0.25, output_dir="detection_test_results"):
    """검증 세트에서 랜덤 샘플을 선택하여 검출 테스트"""
    
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 기존 파일 정리
    for file in output_dir.glob("*"):
        file.unlink()
    
    # 모델 로드
    print(f"📦 모델 로드: {model_path}")
    model = YOLO(model_path)
    
    # 검증 이미지 목록 가져오기
    val_images_dir = Path("data_stage2_20250727/images/val")
    val_images = list(val_images_dir.glob("*.png"))
    
    if not val_images:
        print("❌ 검증 이미지를 찾을 수 없습니다.")
        return
    
    # 랜덤 샘플 선택
    sample_images = random.sample(val_images, min(num_samples, len(val_images)))
    
    print(f"🎲 {len(sample_images)}개 이미지 테스트")
    print(f"📁 결과 저장 위치: {output_dir}")
    
    for img_path in sample_images:
        print(f"\n🖼️  처리 중: {img_path.name}")
        
        # 원본 이미지 복사
        output_image_path = output_dir / img_path.name
        shutil.copy(img_path, output_image_path)
        
        # 검출 실행
        results = model(str(img_path), conf=conf_threshold, save=False)
        
        # txt 파일로 결과 저장 (YOLO 형식)
        txt_path = output_dir / f"{img_path.stem}.txt"
        
        detections = results[0]
        boxes = detections.boxes
        
        if boxes is not None:
            with open(txt_path, 'w') as f:
                for box in boxes:
                    class_id = int(box.cls)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf)
                    
                    # 이미지 크기로 정규화
                    img_height, img_width = detections.orig_shape
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # YOLO 형식: class_id x_center y_center width height confidence
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
            
            print(f"   ✅ {len(boxes)}개 객체 검출됨")
        else:
            # 빈 txt 파일 생성
            txt_path.touch()
            print(f"   ⚠️  검출된 객체 없음")
        
        # 클래스 이름 매핑 파일도 저장 (한 번만)
        if not (output_dir / "classes.txt").exists():
            with open(output_dir / "classes.txt", 'w') as f:
                for idx, name in model.names.items():
                    f.write(f"{idx}: {name}\n")
    
    print(f"\n✅ 완료! 결과 확인:")
    print(f"   - 원본 이미지: {output_dir}/*.png")
    print(f"   - 검출 결과: {output_dir}/*.txt")
    print(f"   - 클래스 매핑: {output_dir}/classes.txt")

def visualize_results(results_dir="detection_test_results"):
    """저장된 결과를 시각화"""
    import cv2
    import numpy as np
    
    results_dir = Path(results_dir)
    output_viz_dir = results_dir / "visualized"
    output_viz_dir.mkdir(exist_ok=True)
    
    # 클래스 이름 로드 (없으면 기본 클래스 매핑 사용)
    class_names = {}
    classes_file = results_dir / "classes.txt"
    
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            for line in f:
                idx, name = line.strip().split(': ')
                class_names[int(idx)] = name
    else:
        print("⚠️  classes.txt 파일이 없습니다. 클래스 번호만 사용합니다.")
        # 기본 Stage 2 클래스 매핑 (필요시)
        for i in range(45):  # Stage 2는 45개 클래스
            class_names[i] = f"class_{i}"
    
    # 클래스별 색상 생성 (HSV 색상 공간 사용)
    num_classes = 45  # Stage 2 클래스 수
    colors = []
    for i in range(num_classes):
        hue = int(180 * i / num_classes)  # HSV에서 Hue는 0-180
        color = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color))
    
    # 각 이미지에 대해 시각화
    for img_path in results_dir.glob("*.png"):
        txt_path = results_dir / f"{img_path.stem}.txt"
        
        if not txt_path.exists():
            continue
        
        # 이미지 로드
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]
        
        # 검출 결과 그리기
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, w, h = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) > 5 else 1.0
                    
                    # 정규화된 좌표를 픽셀로 변환
                    x1 = int((x_center - w/2) * width)
                    y1 = int((y_center - h/2) * height)
                    x2 = int((x_center + w/2) * width)
                    y2 = int((y_center + h/2) * height)
                    
                    # 클래스별 색상 선택
                    color = colors[class_id % len(colors)]
                    
                    # 박스 그리기 (클래스별 색상)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # 클래스 번호만 표시 (같은 색상)
                    label = str(class_id)
                    cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.4, color, 1)
        
        # 저장
        output_path = output_viz_dir / f"viz_{img_path.name}"
        cv2.imwrite(str(output_path), img)
        print(f"시각화 저장: {output_path}")

def test_image_directory(model_path, image_dir, output_dir, conf_threshold=0.25):
    """지정된 디렉토리의 모든 이미지에 대해 검출 테스트"""
    
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 모델 로드
    print(f"📦 모델 로드: {model_path}")
    model = YOLO(model_path)
    
    # 이미지 목록
    image_dir = Path(image_dir)
    image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"❌ {image_dir}에 이미지가 없습니다.")
        return
    
    print(f"📁 {len(image_files)}개 이미지 처리 중...")
    print(f"📁 결과 저장 위치: {output_dir}")
    
    for img_path in image_files:
        print(f"🖼️  처리 중: {img_path.name}")
        
        # 원본 이미지 복사
        output_image_path = output_dir / img_path.name
        shutil.copy(img_path, output_image_path)
        
        # 검출 실행
        results = model(str(img_path), conf=conf_threshold, save=False)
        
        # txt 파일로 결과 저장
        txt_path = output_dir / f"{img_path.stem}.txt"
        
        detections = results[0]
        boxes = detections.boxes
        
        if boxes is not None:
            with open(txt_path, 'w') as f:
                for box in boxes:
                    class_id = int(box.cls)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf)
                    
                    # 이미지 크기로 정규화
                    img_height, img_width = detections.orig_shape
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
        else:
            txt_path.touch()
    
    # 클래스 매핑 파일 저장
    with open(output_dir / "classes.txt", 'w') as f:
        for idx, name in model.names.items():
            f.write(f"{idx}: {name}\n")
    
    print(f"\n✅ 완료!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="간단한 YOLOv8 Detection 테스트")
    parser.add_argument("--model", default="runs/stage2/yolov8s_45classes_0728_0620/weights/best.pt", 
                        help="모델 경로")
    parser.add_argument("--num-samples", type=int, default=5, help="테스트할 이미지 수")
    parser.add_argument("--conf", type=float, default=0.25, help="신뢰도 임계값")
    parser.add_argument("--visualize", action="store_true", help="결과 시각화")
    parser.add_argument("--image-dir", help="테스트할 이미지 디렉토리")
    parser.add_argument("--output", default="detection_test_results", help="출력 디렉토리")
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_results(args.output)
    elif args.image_dir:
        test_image_directory(args.model, args.image_dir, args.output, args.conf)
    else:
        test_detection_simple(args.model, args.num_samples, args.conf, args.output)