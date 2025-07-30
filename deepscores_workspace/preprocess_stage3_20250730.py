#!/usr/bin/env python3
"""
Stage 3 DeepScores 데이터 전처리 스크립트
작성일: 2025-07-30
목적: Stage 2의 클래스 ID 매핑 오류 수정 - clefG 포함, beam/tie/slur 정확히 제외
개선사항:
- clefG (ID: 6) 반드시 포함
- beam (ID: 122), tie (ID: 123), slur (ID: 121) 정확히 제외
- 클래스 이름으로 최종 검증 단계 추가
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
from datetime import datetime

class Stage3Preprocessor:
    def __init__(self, config_file='stage3_preprocess_config.json'):
        # Stage 1 config를 기반으로 수정
        if not Path(config_file).exists():
            # Stage 1 config를 복사하여 수정
            with open('stage1_preprocess_config.json', 'r') as f:
                self.config = json.load(f)
            self.config['target_path'] = './data_stage3_20250730'
            self.config['image_size'] = 1024  # 메모리 절약을 위해 축소
        else:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        
        self.source_path = Path(self.config['source_path'])
        self.target_path = Path(self.config['target_path'])
        
        # 제외할 클래스 ID (cat_id 기준) - Stage 2 오류 수정 + 작은 점들 제외
        self.excluded_classes = {
            '42',   # stem (줄기) - 얇고 검출 어려움
            '2',    # ledgerLine (덧줄) - 얇고 검출 어려움
            '122',  # beam (보) - 복잡한 연결선
            '123',  # tie (붙임줄) - 곡선이고 얇음
            '121',  # slur (이음줄) - 곡선이고 얇음
            '41',   # augmentationDot - 매우 작은 점 (1-2픽셀)
            '73',   # articStaccatoAbove - 매우 작은 점
            '74',   # articStaccatoBelow - 매우 작은 점
            '3'     # repeatDot - 작은 점, recall 낮음
        }
        
        # 로그 시스템 초기화 (먼저)
        self.log_messages = []
        
        # 제외할 클래스를 뺀 나머지만 선택
        original_selected = set(str(cls_id) for cls_id in self.config['selected_classes'])
        self.selected_classes = original_selected - self.excluded_classes
        
        # 클래스 매핑 재생성
        self.regenerate_class_mapping()
        
        # 클래스 이름 검증 (Stage 3 신규 기능)
        self.verify_class_selection()
        
        self.image_size = self.config['image_size']
        self.sample_ratio = self.config['sample_ratio']
        self.max_images = self.config['max_images']
        
        # 출력 디렉토리 생성
        self.setup_directories()
        
        # 로그 파일 경로 설정 (디렉토리 생성 후)
        self.log_file = self.target_path / f'preprocessing_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    def regenerate_class_mapping(self):
        """제외된 클래스를 뺀 새로운 클래스 매핑 생성"""
        sorted_classes = sorted(list(self.selected_classes), key=int)
        self.class_mapping = {cls_id: idx for idx, cls_id in enumerate(sorted_classes)}
        
        print(f"📊 선택된 클래스 수: {len(self.selected_classes)}개 (원래 {len(self.config['selected_classes'])}개에서 {len(self.excluded_classes)}개 제외)")
        self.log(f"Selected classes: {len(self.selected_classes)}, Excluded: {len(self.excluded_classes)} (including small dots)")
    
    def verify_class_selection(self):
        """클래스 선택이 올바른지 이름으로 검증 (Stage 3 신규 기능)"""
        self.log("🔍 클래스 검증 시작...")
        
        # stage1_classes.json에서 클래스 정보 로드
        with open('stage1_classes.json', 'r') as f:
            stage1_info = json.load(f)
        
        # ID -> 이름 매핑 생성
        id_to_name = {}
        for cls in stage1_info['classes']:
            id_to_name[str(cls['id'])] = cls['name']
        
        # 제외된 클래스 검증
        self.log("❌ 제외된 클래스:")
        excluded_names = []
        for class_id in self.excluded_classes:
            class_name = id_to_name.get(class_id, f"Unknown_{class_id}")
            excluded_names.append(class_name)
            self.log(f"   ID {class_id}: {class_name}")
        
        # 포함된 중요 클래스 확인
        self.log("✅ 포함된 중요 클래스:")
        critical_classes = ['clefG', 'clefF', 'clefCAlto', 'noteheadBlack', 'staff']
        for critical in critical_classes:
            found = False
            for class_id in self.selected_classes:
                class_name = id_to_name.get(class_id, "Unknown")
                if critical.lower() in class_name.lower():
                    self.log(f"   ✓ {class_name} (ID: {class_id})")
                    found = True
                    break
            if not found:
                self.log(f"   ⚠️  {critical} - 찾을 수 없음")
        
        # 치명적 오류 검사
        if '6' not in self.selected_classes:
            raise ValueError("❌ CRITICAL: clefG (ID: 6)가 제외되었습니다! 이는 치명적 오류입니다.")
        else:
            self.log("✅ clefG (ID: 6) 정상 포함 확인")
        
        # 잘못 포함된 어려운 클래스 검사
        problematic_in_selection = []
        for class_id in self.selected_classes:
            class_name = id_to_name.get(class_id, "Unknown")
            if any(difficult in class_name.lower() for difficult in ['beam', 'tie', 'slur']):
                problematic_in_selection.append(f"{class_name} (ID: {class_id})")
        
        if problematic_in_selection:
            self.log("⚠️  어려운 클래스가 여전히 포함됨:")
            for cls in problematic_in_selection:
                self.log(f"   - {cls}")
        else:
            self.log("✅ beam, tie, slur 정상 제외 확인")
        
        self.log(f"✅ 클래스 검증 완료 - 총 {len(self.selected_classes)}개 클래스")
    
    def log(self, message):
        """로그 메시지 저장"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_messages.append(log_entry)
        print(f"📝 {message}")
    
    def save_logs(self):
        """로그 파일 저장"""
        with open(self.log_file, 'w') as f:
            f.write('\n'.join(self.log_messages))
    
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
        
        self.log(f"Output directory created: {self.target_path}")
    
    def load_annotations(self, json_file):
        """JSON 어노테이션 로드 및 필터링"""
        self.log(f"Loading annotations from: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 카테고리 정보
        categories = data['categories']
        images = data['images']
        annotations = data['annotations']
        
        # 필터링된 어노테이션만 수집
        filtered_annotations = defaultdict(list)
        annotation_count = 0
        excluded_count = 0
        
        # DeepScores는 annotations도 dict 형태
        if isinstance(annotations, dict):
            for ann_id, ann in annotations.items():
                cat_id = ann.get('cat_id')
                if cat_id:
                    # cat_id가 리스트인 경우 첫 번째 요소 사용
                    if isinstance(cat_id, list):
                        cat_id = cat_id[0] if cat_id else None
                    
                    if cat_id:
                        if cat_id in self.excluded_classes:
                            excluded_count += 1
                            continue
                        
                        if cat_id in self.selected_classes:
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
        
        self.log(f"Filtering complete: {annotation_count:,d} annotations kept, {excluded_count:,d} excluded, {len(filtered_annotations):,d} images")
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
        self.log(f"Processing {split_name} split...")
        
        # 어노테이션 로드
        images, filtered_annotations, categories = self.load_annotations(json_file)
        
        # 이미지 샘플링
        image_list = list(filtered_annotations.keys())
        if len(image_list) > max_images_per_split:
            image_list = random.sample(image_list, max_images_per_split)
        
        self.log(f"Images to process: {len(image_list):,d}")
        
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
        
        self.log(f"{split_name} complete: {processed_count:,d} processed, {skipped_count:,d} skipped")
        return processed_count
    
    def create_yaml_config(self):
        """YOLO 학습을 위한 YAML 설정 파일 생성"""
        # 클래스 이름 로드
        with open('stage1_classes.json', 'r') as f:
            stage1_info = json.load(f)
        
        # 클래스 ID -> 이름 매핑 생성
        id_to_name = {}
        for cls in stage1_info['classes']:
            if str(cls['id']) not in self.excluded_classes:
                id_to_name[str(cls['id'])] = cls['name']
        
        # 새로운 순서대로 클래스 이름 리스트 생성
        sorted_classes = sorted(list(self.selected_classes), key=int)
        class_names = [id_to_name[cls_id] for cls_id in sorted_classes]
        
        yaml_config = {
            'path': str(self.target_path.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': class_names
        }
        
        yaml_path = self.target_path / 'deepscores_stage3.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
        
        self.log(f"YAML config saved: {yaml_path}")
        
        # 제외된 클래스 정보 저장 (Stage 3에서 개선됨)
        excluded_info = {
            'excluded_classes': list(self.excluded_classes),
            'excluded_class_names': [id_to_name.get(cls_id, f"Unknown_{cls_id}") for cls_id in self.excluded_classes],
            'excluded_details': {
                '42': 'stem - 얇고 검출 어려움',
                '2': 'ledgerLine - 얇고 검출 어려움', 
                '122': 'beam - 복잡한 연결선',
                '123': 'tie - 곡선이고 얇음',
                '121': 'slur - 곡선이고 얇음',
                '41': 'augmentationDot - 매우 작은 점 (1-2픽셀)',
                '73': 'articStaccatoAbove - 매우 작은 점',
                '74': 'articStaccatoBelow - 매우 작은 점',
                '3': 'repeatDot - 작은 점, recall 낮음'
            },
            'remaining_classes': len(self.selected_classes),
            'total_classes': len(stage1_info['classes']),
            'critical_classes_included': {
                'clefG': '6' in self.selected_classes,
                'clefF': any('clef' in id_to_name.get(cls_id, '').lower() for cls_id in self.selected_classes),
                'noteheads': any('notehead' in id_to_name.get(cls_id, '').lower() for cls_id in self.selected_classes),
                'staff': any('staff' in id_to_name.get(cls_id, '').lower() for cls_id in self.selected_classes)
            }
        }
        
        with open(self.target_path / 'excluded_classes_info.json', 'w') as f:
            json.dump(excluded_info, f, indent=2)
    
    def run(self):
        """전체 전처리 실행"""
        self.log("🚀 Stage 3 Data Preprocessing Start")
        self.log(f"🎯 주요 개선사항: clefG 포함, beam/tie/slur 정확히 제외")
        self.log(f"Source: {self.source_path}")
        self.log(f"Target: {self.target_path}")
        self.log(f"Selected classes: {len(self.selected_classes)}")
        self.log(f"Excluded classes: {list(self.excluded_classes)}")
        self.log(f"Image size: {self.image_size}x{self.image_size}")
        self.log(f"Sampling ratio: {self.sample_ratio*100:.1f}%")
        
        # 랜덤 시드 설정
        random.seed(42)
        
        # JSON 파일 목록
        json_files = list(self.source_path.glob("deepscores-complete-*.json"))
        if not json_files:
            self.log("❌ No JSON files found!")
            return
        
        self.log(f"Found JSON files: {len(json_files)}")
        
        # 각 분할별 최대 이미지 수 계산
        total_splits = len(json_files)
        max_images_per_split = self.max_images // total_splits
        
        processed_total = 0
        
        # 각 JSON 파일 처리 (train/val 분할)
        for i, json_file in enumerate(json_files):
            split_name = 'train' if i < len(json_files) * 0.8 else 'val'
            processed = self.process_split(json_file, split_name, max_images_per_split)
            processed_total += processed
        
        self.log(f"\n🎉 Preprocessing complete!")
        self.log(f"Total processed images: {processed_total:,d}")
        self.log(f"Output location: {self.target_path}")
        
        # YAML 설정 파일 생성
        self.create_yaml_config()
        
        # 통계 저장
        stats = {
            'total_processed': processed_total,
            'selected_classes': len(self.selected_classes),
            'excluded_classes': list(self.excluded_classes),
            'image_size': self.image_size,
            'sample_ratio': self.sample_ratio,
            'max_images': self.max_images,
            'preprocessing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'stage': 'stage3',
            'improvements_from_stage2': [
                'clefG (ID: 6) 정상 포함',
                'beam (ID: 122) 정확히 제외',
                'tie (ID: 123) 정확히 제외', 
                'slur (ID: 121) 정확히 제외',
                '클래스 이름 검증 단계 추가'
            ]
        }
        
        with open(self.target_path / 'preprocessing_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        # 로그 저장
        self.save_logs()
        self.log(f"Logs saved to: {self.log_file}")

if __name__ == "__main__":
    # Stage 3 전처리 실행
    preprocessor = Stage3Preprocessor()
    preprocessor.run()