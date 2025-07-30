#!/usr/bin/env python3
"""
ScoreEye 마디 이미지 전용 테스트 스크립트
2배 확대 + 하얀색 패딩으로 2048x2048 만들기
"""

import cv2
import numpy as np
from pathlib import Path
from hybrid_music_detector import HybridMusicDetector

def preprocess_scoreeye_image(image_path: str, target_size: int = 2048, scale_factor: float = 0.5) -> np.ndarray:
    """
    ScoreEye 마디 이미지를 YOLOv8에 최적화된 형태로 전처리
    0.5배 축소 후 패딩 추가
    
    Args:
        image_path: 입력 이미지 경로
        target_size: 최종 출력 크기 (정사각형)
        scale_factor: 축소 비율 (0.5 = 50% 크기)
    
    Returns:
        전처리된 이미지 (target_size x target_size)
    """
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    print(f"원본 이미지 크기: {image.shape}")
    
    # 1. 0.7배 축소
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)  # 축소에는 INTER_AREA가 좋음
    
    print(f"{scale_factor}배 축소 후: {scaled.shape}")
    
    # 2. 2048x2048 하얀색 캔버스 생성
    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255  # 하얀색
    
    # 3. 중앙에 배치
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    
    # 이미지가 캔버스보다 큰 경우 잘라내기 (일반적으로는 발생하지 않음)
    end_x = min(start_x + new_width, target_size)
    end_y = min(start_y + new_height, target_size)
    
    crop_width = end_x - start_x
    crop_height = end_y - start_y
    
    canvas[start_y:end_y, start_x:end_x] = scaled[:crop_height, :crop_width]
    
    print(f"최종 캔버스 크기: {canvas.shape}")
    print(f"이미지 배치 위치: ({start_x}, {start_y}) ~ ({end_x}, {end_y})")
    
    return canvas

def test_scoreeye_with_preprocessing(image_path: str):
    """ScoreEye 이미지를 전처리해서 하이브리드 감지 테스트"""
    
    print("=== ScoreEye 마디 이미지 하이브리드 감지 테스트 ===\\n")
    
    # 1. 이미지 전처리
    print("1. 이미지 전처리 중...")
    preprocessed = preprocess_scoreeye_image(image_path)
    
    # 전처리된 이미지 저장
    preprocessed_path = "scoreeye_preprocessed.png"
    cv2.imwrite(preprocessed_path, preprocessed)
    print(f"전처리 결과 저장: {preprocessed_path}")
    
    # 2. 하이브리드 감지기 초기화
    print("\\n2. 하이브리드 감지기 초기화...")
    model_path = "scoreeye-yolov8/stem_fixed_2048_batch22/weights/best.pt"
    detector = HybridMusicDetector(model_path, confidence_threshold=0.3)  # 더 낮은 threshold
    
    # 3. 감지 실행
    print("\\n3. 음악 기호 감지 실행...")
    yolo_detections, stems = detector.detect(preprocessed)
    
    # 4. 결과 출력
    print(f"\\n=== 감지 결과 ===")
    print(f"YOLOv8 감지: {len(yolo_detections)}개")
    print(f"Classical CV stem 감지: {len(stems)}개")
    print(f"총 감지: {len(yolo_detections) + len(stems)}개")
    
    # 클래스별 카운트
    if yolo_detections:
        class_counts = {}
        for det in yolo_detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        print("\\n클래스별 감지 결과:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}개")
    
    if stems:
        print(f"  stem (Classical CV): {len(stems)}개")
    
    # 5. 시각화 저장
    vis_output = "scoreeye_hybrid_detection.png"
    detector.visualize_detections(preprocessed, yolo_detections, stems, vis_output)
    print(f"\\n시각화 결과 저장: {vis_output}")
    
    # 6. JSON 결과 저장
    json_output = "scoreeye_hybrid_detection.json"
    detector.export_results(yolo_detections, stems, json_output)
    print(f"JSON 결과 저장: {json_output}")
    
    return preprocessed, yolo_detections, stems

if __name__ == "__main__":
    # ScoreEye 마디 이미지 테스트
    image_path = "../output/page_01/P1_05_003.png"
    
    try:
        preprocessed, yolo_detections, stems = test_scoreeye_with_preprocessing(image_path)
        print("\\n✅ 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()