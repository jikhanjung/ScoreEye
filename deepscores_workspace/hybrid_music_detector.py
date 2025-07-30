#!/usr/bin/env python3
"""
Hybrid Music Symbol Detector
Combines YOLOv8 (for general symbols) + Classical CV (for stems)

Authors: Claude & User
Date: 2025-07-23
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
from datetime import datetime

@dataclass
class Detection:
    """음악 기호 감지 결과"""
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    detection_method: str  # 'yolo' or 'classical'

@dataclass
class StemDetection(Detection):
    """Stem 감지 결과 (추가 정보 포함)"""
    length: float
    angle: float
    associated_notehead: Optional[int] = None  # 연결된 음표 머리 ID

class ClassicalStemDetector:
    """전통적 컴퓨터 비전을 사용한 Stem 감지"""
    
    def __init__(self):
        self.min_stem_length = 20  # 최소 stem 길이 (픽셀)
        self.max_stem_width = 5    # 최대 stem 너비 (픽셀)
        self.angle_tolerance = 15  # 수직선 각도 허용 오차 (도)
        self.search_radius = 30    # 음표 머리 주변 검색 반경
        
    def preprocess_for_stems(self, image: np.ndarray) -> np.ndarray:
        """Stem 감지를 위한 이미지 전처리"""
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 적응형 임계값으로 이진화
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 수직선 강화를 위한 모폴로지 연산
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        enhanced = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, vertical_kernel)
        
        return enhanced
    
    def detect_vertical_lines(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """HoughLinesP를 사용한 수직선 감지"""
        # Hough Transform 파라미터
        rho = 1
        theta = np.pi / 180
        threshold = int(self.min_stem_length * 0.6)
        min_line_length = self.min_stem_length
        max_line_gap = 5
        
        lines = cv2.HoughLinesP(
            image, rho, theta, threshold, 
            minLineLength=min_line_length, maxLineGap=max_line_gap
        )
        
        if lines is None:
            return []
        
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 수직선 필터링 (각도 체크)
            if x2 - x1 == 0:  # 완전히 수직
                angle = 90
            else:
                angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
            
            # 수직에 가까운 선만 선택
            if angle > 90 - self.angle_tolerance:
                vertical_lines.append((x1, y1, x2, y2))
                
        return vertical_lines
    
    def filter_stems_near_noteheads(
        self, 
        lines: List[Tuple[int, int, int, int]], 
        noteheads: List[Detection]
    ) -> List[StemDetection]:
        """음표 머리 근처의 수직선만 stem으로 분류"""
        stems = []
        
        for line in lines:
            x1, y1, x2, y2 = line
            
            # 선의 중점과 길이 계산
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # 각도 계산
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = np.degrees(np.arctan((y2 - y1) / (x2 - x1)))
            
            # 가장 가까운 음표 머리 찾기
            min_distance = float('inf')
            associated_notehead = None
            
            for i, notehead in enumerate(noteheads):
                # 음표 머리 중심점
                nh_x = (notehead.bbox[0] + notehead.bbox[2]) / 2
                nh_y = (notehead.bbox[1] + notehead.bbox[3]) / 2
                
                # 거리 계산
                distance = np.sqrt((center_x - nh_x)**2 + (center_y - nh_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    associated_notehead = i
            
            # 음표 머리 근처에 있는 경우만 stem으로 인정
            if min_distance <= self.search_radius:
                stem = StemDetection(
                    class_name='stem',
                    confidence=0.9,  # Classical CV는 신뢰도를 고정값으로
                    bbox=[x1 - 1, min(y1, y2), x2 + 1, max(y1, y2)],
                    detection_method='classical',
                    length=length,
                    angle=angle,
                    associated_notehead=associated_notehead
                )
                stems.append(stem)
        
        return stems
    
    def detect_stems(
        self, 
        image: np.ndarray, 
        noteheads: List[Detection]
    ) -> List[StemDetection]:
        """주요 stem 감지 함수"""
        # 전처리
        processed = self.preprocess_for_stems(image)
        
        # 수직선 감지
        lines = self.detect_vertical_lines(processed)
        
        # 음표 머리 근처 필터링
        stems = self.filter_stems_near_noteheads(lines, noteheads)
        
        return stems

class HybridMusicDetector:
    """YOLOv8 + Classical CV 하이브리드 음악 기호 감지기"""
    
    def __init__(self, yolo_model_path: str, confidence_threshold: float = 0.5):
        # YOLOv8 모델 로드
        self.yolo_model = YOLO(yolo_model_path)
        self.confidence_threshold = confidence_threshold
        
        # Classical stem detector 초기화
        self.stem_detector = ClassicalStemDetector()
        
        # 클래스 이름 매핑 (stem 제외)
        self.class_names = {
            0: 'noteheadBlackOnLine',
            2: 'clefG', 
            3: 'restQuarter',
            4: 'beam',
            5: 'augmentationDot',
            6: 'accidentalSharp',
            7: 'accidentalFlat', 
            8: 'accidentalNatural',
            9: 'timeSig4',
            10: 'noteheadHalfOnLine',
            11: 'noteheadWholeOnLine'
        }
        
        # 음표 머리 클래스들
        self.notehead_classes = ['noteheadBlackOnLine', 'noteheadHalfOnLine', 'noteheadWholeOnLine']
    
    def detect_with_yolo(self, image: np.ndarray) -> List[Detection]:
        """YOLOv8를 사용한 기호 감지 (stem 제외)"""
        results = self.yolo_model(image, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 클래스 ID와 신뢰도
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # stem 클래스(ID=1) 제외
                    if class_id == 1:  # stem
                        continue
                        
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # 클래스 이름
                    class_name = self.class_names.get(class_id, f'class_{class_id}')
                    
                    detection = Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=[x1, y1, x2, y2],
                        detection_method='yolo'
                    )
                    detections.append(detection)
        
        return detections
    
    def get_noteheads(self, detections: List[Detection]) -> List[Detection]:
        """감지된 기호 중 음표 머리만 추출"""
        return [d for d in detections if d.class_name in self.notehead_classes]
    
    def detect(self, image: np.ndarray) -> Tuple[List[Detection], List[StemDetection]]:
        """하이브리드 감지: YOLOv8 + Classical CV"""
        
        # Stage 1: YOLOv8로 일반 기호 감지 (stem 제외)
        yolo_detections = self.detect_with_yolo(image)
        
        # Stage 2: 음표 머리 추출
        noteheads = self.get_noteheads(yolo_detections)
        
        # Stage 3: Classical CV로 stem 감지
        stems = self.stem_detector.detect_stems(image, noteheads)
        
        return yolo_detections, stems
    
    def combine_detections(
        self, 
        yolo_detections: List[Detection], 
        stems: List[StemDetection]
    ) -> List[Detection]:
        """모든 감지 결과를 결합"""
        all_detections = yolo_detections.copy()
        all_detections.extend(stems)
        return all_detections
    
    def visualize_detections(
        self, 
        image: np.ndarray, 
        yolo_detections: List[Detection], 
        stems: List[StemDetection], 
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """감지 결과 시각화"""
        vis_image = image.copy()
        
        # YOLOv8 감지 결과 (파란색)
        for det in yolo_detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 라벨 표시
            label = f"{det.class_name} ({det.confidence:.2f})"
            cv2.putText(vis_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Classical CV stem 감지 결과 (빨간색)
        for stem in stems:
            x1, y1, x2, y2 = [int(v) for v in stem.bbox]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 라벨 표시
            label = f"stem (CV, {stem.confidence:.2f})"
            cv2.putText(vis_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            
        return vis_image
    
    def export_results(
        self, 
        yolo_detections: List[Detection], 
        stems: List[StemDetection], 
        output_path: str
    ):
        """결과를 JSON 형식으로 내보내기"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_detections': len(yolo_detections) + len(stems),
            'yolo_detections': len(yolo_detections),
            'classical_stems': len(stems),
            'detections': []
        }
        
        # YOLOv8 감지 결과
        for det in yolo_detections:
            results['detections'].append({
                'class_name': det.class_name,
                'confidence': float(det.confidence),
                'bbox': [float(x) for x in det.bbox],
                'method': det.detection_method
            })
        
        # Classical CV stem 감지 결과
        for stem in stems:
            results['detections'].append({
                'class_name': stem.class_name,
                'confidence': float(stem.confidence),
                'bbox': [float(x) for x in stem.bbox],
                'method': stem.detection_method,
                'length': float(stem.length),
                'angle': float(stem.angle),
                'associated_notehead': int(stem.associated_notehead) if stem.associated_notehead is not None else None
            })
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

def test_hybrid_detector(custom_image=None):
    """하이브리드 감지기 테스트"""
    
    # 모델 경로 설정
    model_path = "scoreeye-yolov8/stem_fixed_2048_batch22/weights/best.pt"
    
    # 테스트 이미지 경로
    if custom_image:
        test_images = [custom_image]
    else:
        test_images = [
            "data/images/val/lg-102414375-aug-beethoven--page-3.png",
            "data/images/val/lg-105569450-aug-beethoven--page-3.png"
        ]
    
    # 하이브리드 감지기 초기화
    detector = HybridMusicDetector(model_path, confidence_threshold=0.5)
    
    for i, img_path in enumerate(test_images):
        if not Path(img_path).exists():
            print(f"이미지를 찾을 수 없습니다: {img_path}")
            continue
            
        print(f"\\n=== 테스트 이미지 {i+1}: {Path(img_path).name} ===")
        
        # 이미지 로드
        image = cv2.imread(img_path)
        if image is None:
            print(f"이미지 로드 실패: {img_path}")
            continue
        
        # 하이브리드 감지 실행
        yolo_detections, stems = detector.detect(image)
        
        # 결과 출력
        print(f"YOLOv8 감지: {len(yolo_detections)}개")
        print(f"Classical CV stem 감지: {len(stems)}개")
        
        # 클래스별 카운트
        class_counts = {}
        for det in yolo_detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
        
        print("\\n클래스별 감지 결과:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}개")
        
        if stems:
            print(f"  stem (Classical CV): {len(stems)}개")
        
        # 시각화 저장
        vis_output = f"hybrid_detection_result_{i+1}.png"
        detector.visualize_detections(image, yolo_detections, stems, vis_output)
        print(f"시각화 결과 저장: {vis_output}")
        
        # JSON 결과 저장
        json_output = f"hybrid_detection_result_{i+1}.json"
        detector.export_results(yolo_detections, stems, json_output)
        print(f"JSON 결과 저장: {json_output}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_hybrid_detector(sys.argv[1])
    else:
        test_hybrid_detector()