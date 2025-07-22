#!/usr/bin/env python3
"""
YOLOv8 기반 악보 기호 탐지 시스템
ScoreEye에서 추출된 마디 이미지에서 개별 음악 기호를 감지합니다.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import argparse
import json

import numpy as np
import cv2
from ultralytics import YOLO
import torch


class SymbolDetector:
    """YOLOv8 기반 음악 기호 감지 클래스"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, device: str = 'auto'):
        """
        초기화
        
        Args:
            model_path: 학습된 YOLOv8 모델 경로 (.pt 파일)
            conf_threshold: 신뢰도 임계값
            device: 사용할 디바이스 ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        # 디바이스 자동 선택
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        
        # 모델 로드
        print(f"🚀 모델 로딩 중... (Device: {self.device})")
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        
        # 클래스 이름 매핑
        self.class_names = self.model.names
        print(f"📋 감지 가능한 클래스 수: {len(self.class_names)}")
        for class_id, class_name in self.class_names.items():
            print(f"   {class_id}: {class_name}")
    
    def detect(self, 
               image_input: Union[str, np.ndarray], 
               return_format: str = 'dict',
               save_visualization: Optional[str] = None) -> List[Dict]:
        """
        이미지에서 악보 기호를 감지합니다.
        
        Args:
            image_input: 이미지 파일 경로 또는 numpy 배열
            return_format: 반환 형식 ('dict', 'yolo_format')  
            save_visualization: 시각화 결과 저장 경로 (선택사항)
            
        Returns:
            감지된 기호들의 리스트. 각 항목은 다음을 포함:
            - box: (x1, y1, x2, y2) 바운딩 박스 좌표
            - class_name: 클래스 이름
            - class_id: 클래스 ID
            - confidence: 신뢰도 점수
            - center: (x_center, y_center) 중심점 좌표
        """
        start_time = time.time()
        
        # 이미지 로드
        if isinstance(image_input, str):
            if not Path(image_input).exists():
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_input}")
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_input}")
        else:
            image = image_input.copy()
        
        # 추론 실행
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        detected_symbols = []
        
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes
                
                for box in boxes:
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    
                    # 신뢰도와 클래스
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    class_name = self.class_names.get(cls_id, f"unknown_{cls_id}")
                    
                    # 중심점 계산
                    x_center = (x1 + x2) // 2
                    y_center = (y1 + y2) // 2
                    
                    symbol_data = {
                        'box': (x1, y1, x2, y2),
                        'class_name': class_name,
                        'class_id': cls_id,
                        'confidence': conf,
                        'center': (x_center, y_center),
                        'width': x2 - x1,
                        'height': y2 - y1
                    }
                    
                    detected_symbols.append(symbol_data)
        
        # 신뢰도순으로 정렬
        detected_symbols.sort(key=lambda x: x['confidence'], reverse=True)
        
        # 처리 시간 계산
        processing_time = time.time() - start_time
        
        # 시각화 저장
        if save_visualization:
            self._save_visualization(image, detected_symbols, save_visualization)
        
        # 처리 정보 출력
        print(f"⚡ 감지 완료: {len(detected_symbols)}개 기호 ({processing_time:.3f}초)")
        
        return detected_symbols
    
    def detect_batch(self, 
                     image_paths: List[str], 
                     output_dir: Optional[str] = None,
                     save_visualizations: bool = False) -> Dict[str, List[Dict]]:
        """
        여러 이미지에 대해 배치 감지 수행
        
        Args:
            image_paths: 이미지 파일 경로들의 리스트
            output_dir: 결과 저장 디렉터리
            save_visualizations: 시각화 이미지 저장 여부
            
        Returns:
            각 이미지별 감지 결과 딕셔너리
        """
        results = {}
        
        print(f"📦 배치 감지 시작: {len(image_paths)}개 이미지")
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if save_visualizations:
                vis_dir = output_dir / "visualizations"
                vis_dir.mkdir(exist_ok=True)
        
        for i, image_path in enumerate(image_paths, 1):
            image_path = Path(image_path)
            print(f"[{i}/{len(image_paths)}] 처리 중: {image_path.name}")
            
            try:
                # 시각화 경로 설정
                vis_path = None
                if save_visualizations and output_dir:
                    vis_path = vis_dir / f"{image_path.stem}_detected.jpg"
                
                # 감지 실행
                detected = self.detect(str(image_path), save_visualization=vis_path)
                results[str(image_path)] = detected
                
                # JSON 결과 저장
                if output_dir:
                    json_path = output_dir / f"{image_path.stem}_detections.json"
                    with open(json_path, 'w') as f:
                        json.dump(detected, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                print(f"   ❌ 오류: {str(e)}")
                results[str(image_path)] = []
        
        print(f"✅ 배치 감지 완료!")
        return results
    
    def tiled_inference(self, 
                       image: np.ndarray, 
                       tile_size: int = 1024, 
                       overlap: int = 128) -> List[Dict]:
        """
        큰 이미지를 타일로 나누어 추론하고 결과를 병합
        
        Args:
            image: 입력 이미지 (numpy 배열)
            tile_size: 타일 크기 (정사각형)
            overlap: 타일 간 겹침 크기
            
        Returns:
            전체 이미지에서의 감지 결과
        """
        h, w = image.shape[:2]
        
        # 타일이 필요없을 정도로 작은 이미지면 일반 추론
        if max(h, w) <= tile_size:
            return self.detect(image)
        
        print(f"🔲 Tiled Inference: {w}x{h} → {tile_size}x{tile_size} 타일")
        
        step = tile_size - overlap
        all_detections = []
        
        for y in range(0, h - tile_size + 1, step):
            for x in range(0, w - tile_size + 1, step):
                # 타일 추출
                tile = image[y:y+tile_size, x:x+tile_size]
                
                # 타일에서 감지 실행
                tile_detections = self.detect(tile)
                
                # 전체 이미지 좌표계로 변환
                for detection in tile_detections:
                    x1, y1, x2, y2 = detection['box']
                    detection['box'] = (x1+x, y1+y, x2+x, y2+y)
                    detection['center'] = (detection['center'][0]+x, detection['center'][1]+y)
                
                all_detections.extend(tile_detections)
        
        # 경계 영역 처리 (오른쪽 및 아래쪽 끝)
        if w % step != 0:
            for y in range(0, h - tile_size + 1, step):
                x = w - tile_size
                tile = image[y:y+tile_size, x:x+tile_size]
                tile_detections = self.detect(tile)
                
                for detection in tile_detections:
                    x1, y1, x2, y2 = detection['box']
                    detection['box'] = (x1+x, y1+y, x2+x, y2+y)
                    detection['center'] = (detection['center'][0]+x, detection['center'][1]+y)
                
                all_detections.extend(tile_detections)
        
        if h % step != 0:
            y = h - tile_size
            for x in range(0, w - tile_size + 1, step):
                tile = image[y:y+tile_size, x:x+tile_size]
                tile_detections = self.detect(tile)
                
                for detection in tile_detections:
                    x1, y1, x2, y2 = detection['box']
                    detection['box'] = (x1+x, y1+y, x2+x, y2+y)
                    detection['center'] = (detection['center'][0]+x, detection['center'][1]+y)
                
                all_detections.extend(tile_detections)
        
        # 중복 제거 (NMS 유사한 방식)
        final_detections = self._remove_duplicate_detections(all_detections)
        
        print(f"   타일 감지: {len(all_detections)}개 → 중복 제거 후: {len(final_detections)}개")
        
        return final_detections
    
    def _remove_duplicate_detections(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """중복 감지 결과 제거 (NMS 방식)"""
        if len(detections) <= 1:
            return detections
        
        # 신뢰도순으로 정렬
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # 남은 detection들과 IoU 계산
            remaining = []
            for det in detections:
                iou = self._calculate_iou(current['box'], det['box'])
                
                # 같은 클래스이고 IoU가 임계값보다 낮으면 유지
                if current['class_id'] != det['class_id'] or iou < iou_threshold:
                    remaining.append(det)
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """두 바운딩 박스간 IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교집합 영역 계산
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 합집합 영역 계산
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _save_visualization(self, image: np.ndarray, detections: List[Dict], output_path: str):
        """감지 결과 시각화 저장"""
        vis_image = image.copy()
        
        # 색상 팔레트 (클래스별 다른 색상)
        colors = [
            (255, 0, 0),    # 빨강
            (0, 255, 0),    # 초록
            (0, 0, 255),    # 파랑
            (255, 255, 0),  # 노랑
            (255, 0, 255),  # 자주
            (0, 255, 255),  # 청록
            (128, 0, 128),  # 보라
            (255, 165, 0),  # 주황
            (0, 128, 0),    # 어두운 초록
            (128, 128, 0),  # 올리브
        ]
        
        for detection in detections:
            x1, y1, x2, y2 = detection['box']
            class_name = detection['class_name']
            confidence = detection['confidence']
            class_id = detection['class_id']
            
            # 클래스별 색상 선택
            color = colors[class_id % len(colors)]
            
            # 바운딩 박스 그리기
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # 라벨 텍스트
            label = f"{class_name} {confidence:.2f}"
            
            # 텍스트 배경 크기 계산
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 텍스트 배경 그리기
            cv2.rectangle(
                vis_image, 
                (x1, y1 - text_height - baseline - 5), 
                (x1 + text_width, y1), 
                color, -1
            )
            
            # 텍스트 그리기
            cv2.putText(
                vis_image, label, (x1, y1 - baseline - 3), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        # 저장
        cv2.imwrite(str(output_path), vis_image)
    
    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            'model_path': str(self.model_path),
            'device': self.device,
            'class_count': len(self.class_names),
            'class_names': self.class_names,
            'confidence_threshold': self.conf_threshold
        }


def main():
    """메인 실행 함수 (테스트용)"""
    parser = argparse.ArgumentParser(description="YOLOv8 음악 기호 감지")
    parser.add_argument("--model", required=True, help="YOLOv8 모델 파일 (.pt)")
    parser.add_argument("--image", help="단일 이미지 파일")
    parser.add_argument("--batch", help="이미지 디렉터리 (배치 처리)")
    parser.add_argument("--output", default="output", help="결과 저장 디렉터리")
    parser.add_argument("--conf", type=float, default=0.5, help="신뢰도 임계값")
    parser.add_argument("--device", default="auto", help="디바이스 (auto/cpu/cuda/mps)")
    parser.add_argument("--tiled", action="store_true", help="Tiled inference 사용")
    parser.add_argument("--tile-size", type=int, default=1024, help="타일 크기")
    parser.add_argument("--visualize", action="store_true", help="시각화 저장")
    
    args = parser.parse_args()
    
    # 모델 초기화
    detector = SymbolDetector(args.model, conf_threshold=args.conf, device=args.device)
    
    # 모델 정보 출력
    model_info = detector.get_model_info()
    print("🎯 모델 정보:")
    for key, value in model_info.items():
        if key != 'class_names':
            print(f"   {key}: {value}")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if args.image:
            # 단일 이미지 처리
            print(f"\n🖼️ 단일 이미지 처리: {args.image}")
            
            image = cv2.imread(args.image)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {args.image}")
            
            vis_path = output_dir / "detection_result.jpg" if args.visualize else None
            
            if args.tiled:
                detections = detector.tiled_inference(image, tile_size=args.tile_size)
            else:
                detections = detector.detect(image, save_visualization=vis_path)
            
            # 결과 저장
            result_path = output_dir / "detections.json"
            with open(result_path, 'w') as f:
                json.dump(detections, f, indent=2, ensure_ascii=False)
            
            print(f"📄 결과 저장: {result_path}")
            if vis_path:
                print(f"🖼️ 시각화 저장: {vis_path}")
            
        elif args.batch:
            # 배치 처리
            batch_dir = Path(args.batch)
            if not batch_dir.exists():
                raise FileNotFoundError(f"배치 디렉터리를 찾을 수 없습니다: {batch_dir}")
            
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(batch_dir.glob(ext))
            
            if not image_files:
                raise ValueError(f"처리할 이미지가 없습니다: {batch_dir}")
            
            print(f"\n📦 배치 처리: {len(image_files)}개 파일")
            
            results = detector.detect_batch(
                [str(f) for f in image_files],
                output_dir=output_dir,
                save_visualizations=args.visualize
            )
            
            # 전체 결과 요약
            total_detections = sum(len(detections) for detections in results.values())
            print(f"📊 전체 감지된 기호 수: {total_detections}")
            
        else:
            print("❌ --image 또는 --batch 옵션을 지정해주세요.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()