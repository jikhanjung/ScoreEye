#!/usr/bin/env python3
"""
ScoreEye 마디 이미지 스케일 최적화 테스트
0.1부터 1.0까지 0.1 단위로 테스트
"""

import cv2
import numpy as np
from pathlib import Path
from hybrid_music_detector import HybridMusicDetector
import json

def preprocess_with_scale(image_path: str, scale_factor: float, target_size: int = 2048) -> np.ndarray:
    """특정 스케일로 이미지 전처리"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
    
    # 스케일 적용
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    if scale_factor < 1.0:
        # 축소
        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        # 확대 (1.0은 원본)
        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # 2048x2048 하얀색 캔버스에 중앙 배치
    canvas = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    
    end_x = min(start_x + new_width, target_size)
    end_y = min(start_y + new_height, target_size)
    
    crop_width = end_x - start_x
    crop_height = end_y - start_y
    
    canvas[start_y:end_y, start_x:end_x] = scaled[:crop_height, :crop_width]
    
    return canvas

def test_scale_optimization(image_path: str):
    """0.1부터 1.0까지 스케일 최적화 테스트"""
    
    print("=== ScoreEye 스케일 최적화 테스트 ===\\n")
    
    # 하이브리드 감지기 초기화
    model_path = "scoreeye-yolov8/stem_fixed_2048_batch22/weights/best.pt"
    detector = HybridMusicDetector(model_path, confidence_threshold=0.3)
    
    # 결과 저장용
    results = []
    
    # 0.3부터 1.0까지 0.1 단위로 테스트
    for i in range(3, 11):  # 3부터 10까지 (0.3부터 1.0)
        scale = i * 0.1
        
        print(f"\\n--- 스케일 {scale:.1f} 테스트 ---")
        
        try:
            # 이미지 전처리
            preprocessed = preprocess_with_scale(image_path, scale)
            
            # 원본 크기 계산
            original = cv2.imread(image_path)
            orig_h, orig_w = original.shape[:2]
            scaled_h, scaled_w = int(orig_h * scale), int(orig_w * scale)
            
            print(f"원본: {orig_h}×{orig_w} → 스케일링: {scaled_h}×{scaled_w}")
            
            # 감지 실행
            yolo_detections, stems = detector.detect(preprocessed)
            
            # 클래스별 카운트
            class_counts = {}
            total_confidence = 0
            max_confidence = 0
            
            for det in yolo_detections:
                class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
                total_confidence += det.confidence
                max_confidence = max(max_confidence, det.confidence)
            
            if stems:
                class_counts['stem'] = len(stems)
            
            # 평균 신뢰도 계산
            avg_confidence = total_confidence / len(yolo_detections) if yolo_detections else 0
            
            # 결과 출력
            total_detections = len(yolo_detections) + len(stems)
            print(f"총 감지: {total_detections}개")
            print(f"평균 신뢰도: {avg_confidence:.3f}")
            print(f"최고 신뢰도: {max_confidence:.3f}")
            
            if class_counts:
                print("감지된 클래스:")
                for class_name, count in sorted(class_counts.items()):
                    print(f"  {class_name}: {count}개")
            else:
                print("감지된 기호 없음")
            
            # 결과 저장
            result = {
                'scale': scale,
                'scaled_size': f"{scaled_h}×{scaled_w}",
                'total_detections': total_detections,
                'yolo_detections': len(yolo_detections),
                'stem_detections': len(stems),
                'avg_confidence': round(avg_confidence, 3),
                'max_confidence': round(max_confidence, 3),
                'class_counts': class_counts,
                'unique_classes': len(class_counts)
            }
            results.append(result)
            
            # 시각화 저장 (최고 성능 몇 개만)
            if total_detections > 0 and (scale in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
                vis_output = f"scale_{scale:.1f}_detection.png"
                detector.visualize_detections(preprocessed, yolo_detections, stems, vis_output)
                print(f"시각화 저장: {vis_output}")
            
        except Exception as e:
            print(f"스케일 {scale:.1f} 테스트 실패: {e}")
            results.append({
                'scale': scale,
                'error': str(e)
            })
    
    # 최종 결과 분석
    print("\\n\\n=== 전체 결과 요약 ===")
    print("스케일 | 크기       | 총감지 | YOLO | Stem | 평균신뢰도 | 최고신뢰도 | 클래스수")
    print("-" * 85)
    
    valid_results = [r for r in results if 'error' not in r]
    
    for result in valid_results:
        scale = result['scale']
        size = result['scaled_size']
        total = result['total_detections']
        yolo = result['yolo_detections']
        stem = result['stem_detections']
        avg_conf = result['avg_confidence']
        max_conf = result['max_confidence']
        classes = result['unique_classes']
        
        print(f"{scale:5.1f} | {size:10s} | {total:6d} | {yolo:4d} | {stem:4d} | {avg_conf:8.3f} | {max_conf:8.3f} | {classes:6d}")
    
    # 최고 성능 찾기
    if valid_results:
        # 다양한 기준으로 최고 성능 찾기
        best_total = max(valid_results, key=lambda x: x['total_detections'])
        best_confidence = max(valid_results, key=lambda x: x['avg_confidence'])
        best_variety = max(valid_results, key=lambda x: x['unique_classes'])
        
        print(f"\\n=== 최고 성능 분석 ===")
        print(f"최다 감지: 스케일 {best_total['scale']:.1f} ({best_total['total_detections']}개)")
        print(f"최고 신뢰도: 스케일 {best_confidence['scale']:.1f} ({best_confidence['avg_confidence']:.3f})")
        print(f"최다 클래스: 스케일 {best_variety['scale']:.1f} ({best_variety['unique_classes']}개)")
        
        # 종합 점수 계산 (정규화된 가중 합)
        max_detections = max(r['total_detections'] for r in valid_results)
        max_conf = max(r['avg_confidence'] for r in valid_results)
        max_classes = max(r['unique_classes'] for r in valid_results)
        
        for result in valid_results:
            # 0-1로 정규화
            norm_detections = result['total_detections'] / max_detections if max_detections > 0 else 0
            norm_confidence = result['avg_confidence'] / max_conf if max_conf > 0 else 0
            norm_classes = result['unique_classes'] / max_classes if max_classes > 0 else 0
            
            # 가중 합 (감지수 40%, 신뢰도 40%, 다양성 20%)
            composite_score = norm_detections * 0.4 + norm_confidence * 0.4 + norm_classes * 0.2
            result['composite_score'] = round(composite_score, 3)
        
        best_overall = max(valid_results, key=lambda x: x['composite_score'])
        print(f"\\n🎯 종합 최적 스케일: {best_overall['scale']:.1f} (종합점수: {best_overall['composite_score']:.3f})")
    
    # JSON으로 결과 저장
    with open('scale_optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\\n상세 결과 저장: scale_optimization_results.json")
    
    return results

if __name__ == "__main__":
    image_path = "../output/page_01/P1_05_001.png"
    
    try:
        results = test_scale_optimization(image_path)
        print("\\n✅ 스케일 최적화 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()