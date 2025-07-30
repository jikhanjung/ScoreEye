#!/usr/bin/env python3
"""
Stage 1 DeepScores 데이터 전처리 스크립트
50개 핵심 클래스로 제한한 데이터셋 생성
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
import random
from collections import defaultdict
import yaml
from tqdm import tqdm
import shutil

class Stage1Preprocessor:
    def __init__(self, config_file='stage1_preprocess_config.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.source_path = Path(self.config['source_path'])
        self.target_path = Path(self.config['target_path'])
        # 문자열로 변환하여 JSON의 cat_id와 타입 일치
        self.selected_classes = set(str(cls_id) for cls_id in self.config['selected_classes'])
        self.class_mapping = self.config['class_mapping']
        self.image_size = self.config['image_size']
        self.sample_ratio = self.config['sample_ratio']
        self.max_images = self.config['max_images']
        
        # 출력 디렉토리 생성
        self.setup_directories()
    
    def setup_directories(self):
        """데이터셋 디렉토리 구조 생성"""
        dirs = [
            'images/train',
            'images/val',
            'labels/train',
            'labels/val'
        ]
        
        for dir_name in dirs:
            (self.target_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        print(f"📁 출력 디렉토리 생성: {self.target_path}")
    
    def load_annotations(self, json_file):
        """JSON 어노테이션 로드 및 필터링"""
        print(f"📖 어노테이션 로드 중: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 카테고리 정보
        categories = data['categories']
        images = data['images']
        annotations = data['annotations']
        
        # 필터링된 어노테이션만 수집
        filtered_annotations = defaultdict(list)
        annotation_count = 0
        
        # DeepScores는 annotations도 dict 형태
        if isinstance(annotations, dict):
            for ann_id, ann in annotations.items():
                cat_id = ann.get('cat_id')
                if cat_id:
                    # cat_id가 리스트인 경우 첫 번째 요소 사용
                    if isinstance(cat_id, list):
                        cat_id = cat_id[0] if cat_id else None
                    
                    if cat_id and cat_id in self.selected_classes:
                        # DeepScores는 img_id를 사용
                        image_id = ann.get('img_id') or ann.get('image_id')
                        # DeepScores는 a_bbox를 사용 (axis-aligned bbox)
                        bbox = ann.get('a_bbox') or ann.get('bbox', [0, 0, 1, 1])
                        
                        if image_id and bbox:
                            filtered_annotations[image_id].append({
                                'cat_id': cat_id,
                                'bbox': bbox,
                                'area': ann.get('area', 1)
                            })
                            annotation_count += 1
        
        print(f"✅ 필터링 완료: {annotation_count:,d}개 어노테이션, {len(filtered_annotations):,d}개 이미지")
        return images, filtered_annotations, categories
    
    def convert_bbox_to_yolo(self, bbox, img_width, img_height):
        """DeepScores bbox를 YOLO 형식으로 변환
        DeepScores의 a_bbox는 [x1, y1, x2, y2] 형식
        """
        x1, y1, x2, y2 = bbox
        
        # width, height 계산
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # 중심점 좌표 계산
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 정규화
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height
        
        # 좌표가 0-1 범위를 벗어나지 않도록 클리핑
        center_x = max(0.0, min(1.0, center_x))
        center_y = max(0.0, min(1.0, center_y))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))
        
        return [center_x, center_y, width, height]
    
    def process_image(self, image_path, annotations, output_image_path, output_label_path):
        """단일 이미지 처리"""
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            return False
        
        orig_height, orig_width = image.shape[:2]
        
        # 이미지 리사이즈 (정사각형으로)
        if orig_width != self.image_size or orig_height != self.image_size:
            # 가로세로 비율 유지하면서 패딩 추가
            scale = self.image_size / max(orig_width, orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # 리사이즈
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # 패딩 추가 (가운데 정렬)
            pad_x = (self.image_size - new_width) // 2
            pad_y = (self.image_size - new_height) // 2
            
            processed_image = np.full((self.image_size, self.image_size, 3), 255, dtype=np.uint8)
            processed_image[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
        else:
            processed_image = image.copy()
            scale = 1.0
            pad_x = pad_y = 0
        
        # 이미지 저장
        cv2.imwrite(str(output_image_path), processed_image)
        
        # 라벨 변환 및 저장
        yolo_labels = []
        for ann in annotations:
            cat_id = ann['cat_id']
            bbox = ann['bbox']
            
            # 바운딩 박스를 새로운 이미지 크기에 맞게 조정
            # DeepScores bbox는 [x1, y1, x2, y2] 형식
            x1, y1, x2, y2 = bbox
            
            # 원본 이미지 범위를 벗어나는 바운딩 박스 필터링
            if (x1 < 0 or y1 < 0 or x2 > orig_width or y2 > orig_height or
                x1 >= x2 or y1 >= y2):
                continue
            
            # 스케일링 적용
            x1_scaled = x1 * scale
            y1_scaled = y1 * scale
            x2_scaled = x2 * scale
            y2_scaled = y2 * scale
            
            # 패딩 오프셋 적용
            x1_final = x1_scaled + pad_x
            y1_final = y1_scaled + pad_y
            x2_final = x2_scaled + pad_x
            y2_final = y2_scaled + pad_y
            
            # 변환된 이미지 범위를 벗어나는 바운딩 박스 필터링
            if (x1_final < 0 or y1_final < 0 or 
                x2_final > self.image_size or y2_final > self.image_size):
                continue
            
            # 너무 작은 바운딩 박스 필터링 (1픽셀 이하)
            width_final = x2_final - x1_final
            height_final = y2_final - y1_final
            if width_final < 1 or height_final < 1:
                continue
            
            # YOLO 형식으로 변환
            yolo_bbox = self.convert_bbox_to_yolo([x1_final, y1_final, x2_final, y2_final], 
                                                 self.image_size, self.image_size)
            
            # 추가 검증: 모든 좌표가 유효한 범위인지 확인
            if all(0.0 <= coord <= 1.0 for coord in yolo_bbox):
                # 클래스 ID 매핑 (문자열로 변환)
                cat_id_str = str(cat_id)
                if cat_id_str in self.class_mapping:
                    class_id = self.class_mapping[cat_id_str]
                    yolo_labels.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")
        
        # 라벨 파일 저장
        if yolo_labels:
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(yolo_labels))
        else:
            # 빈 라벨 파일 생성
            output_label_path.touch()
        
        return True
    
    def process_split(self, json_file, split_name, max_images_per_split):
        """데이터 분할 처리"""
        print(f"\n🔄 {split_name} 데이터 처리 중...")
        
        # 어노테이션 로드
        images, filtered_annotations, categories = self.load_annotations(json_file)
        
        # 이미지 샘플링
        image_list = list(filtered_annotations.keys())
        if len(image_list) > max_images_per_split:
            image_list = random.sample(image_list, max_images_per_split)
        
        print(f"📊 처리할 이미지 수: {len(image_list):,d}")
        
        # 출력 경로 설정
        image_output_dir = self.target_path / 'images' / split_name
        label_output_dir = self.target_path / 'labels' / split_name
        
        processed_count = 0
        skipped_count = 0
        
        # 이미지 처리
        for image_id in tqdm(image_list, desc=f"Processing {split_name}"):
            # 이미지 정보 찾기 (DeepScores는 문자열 키 사용)
            image_info = None
            if isinstance(images, dict):
                # 문자열과 정수 둘 다 시도
                image_info = images.get(str(image_id)) or images.get(image_id)
            else:
                for img in images:
                    if img.get('id') == image_id or str(img.get('id')) == str(image_id):
                        image_info = img
                        break
            
            if not image_info:
                skipped_count += 1
                continue
            
            # 파일 경로
            filename = image_info.get('filename') or image_info.get('file_name', f"{image_id}.png")
            image_path = self.source_path / 'images' / filename
            
            if not image_path.exists():
                skipped_count += 1
                continue
            
            # 출력 경로
            output_image_path = image_output_dir / filename
            output_label_path = label_output_dir / (Path(filename).stem + '.txt')
            
            # 이미지 처리
            if self.process_image(image_path, filtered_annotations[image_id], 
                                 output_image_path, output_label_path):
                processed_count += 1
            else:
                skipped_count += 1
        
        print(f"✅ {split_name} 완료: {processed_count:,d}개 처리, {skipped_count:,d}개 스킵")
        return processed_count
    
    def run(self):
        """전체 전처리 실행"""
        print("🚀 Stage 1 데이터 전처리 시작")
        print(f"📁 소스: {self.source_path}")
        print(f"📁 타겟: {self.target_path}")
        print(f"🎯 선택된 클래스: {len(self.selected_classes)}개")
        print(f"📏 이미지 크기: {self.image_size}x{self.image_size}")
        print(f"📊 샘플링 비율: {self.sample_ratio*100:.1f}%")
        
        # 랜덤 시드 설정
        random.seed(42)
        
        # JSON 파일 목록
        json_files = list(self.source_path.glob("deepscores-complete-*.json"))
        if not json_files:
            print("❌ JSON 파일을 찾을 수 없습니다!")
            return
        
        print(f"📄 발견된 JSON 파일: {len(json_files)}개")
        
        # 각 분할별 최대 이미지 수 계산
        total_splits = len(json_files)
        max_images_per_split = self.max_images // total_splits
        
        processed_total = 0
        
        # 각 JSON 파일 처리 (train/val 분할)
        for i, json_file in enumerate(json_files):
            split_name = 'train' if i < len(json_files) * 0.8 else 'val'
            processed = self.process_split(json_file, split_name, max_images_per_split)
            processed_total += processed
        
        print(f"\n🎉 전처리 완료!")
        print(f"📊 총 처리된 이미지: {processed_total:,d}개")
        print(f"💾 출력 위치: {self.target_path}")
        
        # 통계 저장
        stats = {
            'total_processed': processed_total,
            'selected_classes': len(self.selected_classes),
            'image_size': self.image_size,
            'sample_ratio': self.sample_ratio,
            'max_images': self.max_images
        }
        
        with open(self.target_path / 'preprocessing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    # 전처리 실행
    preprocessor = Stage1Preprocessor()
    preprocessor.run()