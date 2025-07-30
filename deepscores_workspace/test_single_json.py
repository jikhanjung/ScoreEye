#!/usr/bin/env python3
"""
단일 JSON 파일로 전처리 테스트
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
import random
from collections import defaultdict
from tqdm import tqdm
import shutil

class SingleJSONProcessor:
    def __init__(self):
        # 설정 로드
        with open('stage1_preprocess_config.json', 'r') as f:
            config = json.load(f)
        
        self.source_path = Path(config['source_path'])
        self.target_path = Path('./test_data')
        self.selected_classes = set(str(cls_id) for cls_id in config['selected_classes'])
        self.class_mapping = config['class_mapping']
        self.image_size = config['image_size']
        
        # 출력 디렉토리 생성
        self.setup_directories()
    
    def setup_directories(self):
        """테스트 디렉토리 구조 생성"""
        if self.target_path.exists():
            shutil.rmtree(self.target_path)
        
        dirs = ['images', 'labels']
        for dir_name in dirs:
            (self.target_path / dir_name).mkdir(parents=True, exist_ok=True)
    
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
    
    def process_single_json(self, json_file, max_images=10):
        """단일 JSON 파일 처리"""
        print(f"📖 처리 중: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        categories = data['categories']
        images = data['images']
        annotations = data['annotations']
        
        # 필터링된 어노테이션 수집
        filtered_annotations = defaultdict(list)
        annotation_count = 0
        
        if isinstance(annotations, dict):
            for ann_id, ann in annotations.items():
                cat_id = ann.get('cat_id')
                if cat_id:
                    if isinstance(cat_id, list):
                        cat_id = cat_id[0] if cat_id else None
                    
                    if cat_id and cat_id in self.selected_classes:
                        image_id = ann.get('img_id') or ann.get('image_id')
                        bbox = ann.get('a_bbox') or ann.get('bbox', [0, 0, 1, 1])
                        
                        if image_id and bbox:
                            filtered_annotations[image_id].append({
                                'cat_id': cat_id,
                                'bbox': bbox,
                                'area': ann.get('area', 1)
                            })
                            annotation_count += 1
        
        print(f"✅ 필터링 완료: {annotation_count:,d}개 어노테이션, {len(filtered_annotations):,d}개 이미지")
        
        # 처음 N개 이미지만 처리
        image_list = list(filtered_annotations.keys())[:max_images]
        
        processed_count = 0
        error_count = 0
        bbox_errors = []
        
        for image_id in tqdm(image_list, desc="Processing images"):
            print(f"\n🔍 이미지 ID 검색: {image_id} (타입: {type(image_id)})")
            
            # 이미지 정보 찾기
            image_info = None
            if isinstance(images, dict):
                print(f"   images는 dict 타입, 키 샘플: {list(images.keys())[:3]}")
                image_info = images.get(str(image_id)) or images.get(image_id)
                if not image_info:
                    # 모든 키를 확인해보기
                    for key, value in list(images.items())[:3]:
                        print(f"   샘플 키: {key} (타입: {type(key)}) -> {value.get('filename', 'no filename')}")
            else:
                print(f"   images는 list 타입, 길이: {len(images)}")
                for img in images[:3]:
                    img_id = img.get('id')
                    print(f"   샘플 ID: {img_id} (타입: {type(img_id)}) -> {img.get('filename', 'no filename')}")
                    if img_id == image_id or str(img_id) == str(image_id):
                        image_info = img
                        break
            
            if image_info:
                print(f"   ✅ 이미지 정보 발견: {image_info.get('filename', 'no filename')}")
            else:
                print(f"   ❌ 이미지 정보 없음")
                error_count += 1
                continue
            
            # 파일 경로
            filename = image_info.get('filename') or image_info.get('file_name', f"{image_id}.png")
            image_path = self.source_path / 'images' / filename
            
            if not image_path.exists():
                print(f"❌ 이미지 파일 없음: {image_path}")
                error_count += 1
                continue
            
            # 이미지 로드
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ 이미지 로드 실패: {image_path}")
                error_count += 1
                continue
            
            orig_height, orig_width = image.shape[:2]
            print(f"\n🖼️  이미지: {filename}")
            print(f"   원본 크기: {orig_width}x{orig_height}")
            
            # 이미지 리사이즈
            scale = self.image_size / max(orig_width, orig_height)
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # 패딩 추가
            pad_x = (self.image_size - new_width) // 2
            pad_y = (self.image_size - new_height) // 2
            
            processed_image = np.full((self.image_size, self.image_size, 3), 255, dtype=np.uint8)
            processed_image[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized
            
            print(f"   스케일: {scale:.3f}, 새 크기: {new_width}x{new_height}")
            print(f"   패딩: ({pad_x}, {pad_y})")
            
            # 바운딩 박스 처리
            yolo_labels = []
            valid_boxes = 0
            invalid_boxes = 0
            
            for ann in filtered_annotations[image_id]:
                cat_id = ann['cat_id']
                bbox = ann['bbox']
                x1, y1, x2, y2 = bbox
                
                print(f"   📦 원본 bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
                
                # 원본 이미지 범위 검사
                if (x1 < 0 or y1 < 0 or x2 > orig_width or y2 > orig_height or
                    x1 >= x2 or y1 >= y2):
                    print(f"      ❌ 원본 범위 초과")
                    invalid_boxes += 1
                    bbox_errors.append({
                        'image': filename,
                        'original_bbox': bbox,
                        'image_size': (orig_width, orig_height),
                        'error': 'original_bounds'
                    })
                    continue
                
                # 스케일링 적용
                x1_scaled = x1 * scale
                y1_scaled = y1 * scale
                x2_scaled = x2 * scale
                y2_scaled = y2 * scale
                
                # 패딩 적용
                x1_final = x1_scaled + pad_x
                y1_final = y1_scaled + pad_y
                x2_final = x2_scaled + pad_x
                y2_final = y2_scaled + pad_y
                
                print(f"      스케일링 후: [{x1_scaled:.1f}, {y1_scaled:.1f}, {x2_scaled:.1f}, {y2_scaled:.1f}]")
                print(f"      패딩 후: [{x1_final:.1f}, {y1_final:.1f}, {x2_final:.1f}, {y2_final:.1f}]")
                
                # 최종 범위 검사
                if (x1_final < 0 or y1_final < 0 or 
                    x2_final > self.image_size or y2_final > self.image_size):
                    print(f"      ❌ 최종 범위 초과")
                    invalid_boxes += 1
                    bbox_errors.append({
                        'image': filename,
                        'final_bbox': [x1_final, y1_final, x2_final, y2_final],
                        'image_size': (self.image_size, self.image_size),
                        'error': 'final_bounds'
                    })
                    continue
                
                # 크기 검사
                width_final = x2_final - x1_final
                height_final = y2_final - y1_final
                if width_final < 1 or height_final < 1:
                    print(f"      ❌ 너무 작음")
                    invalid_boxes += 1
                    continue
                
                # YOLO 변환
                yolo_bbox = self.convert_bbox_to_yolo([x1_final, y1_final, x2_final, y2_final], 
                                                     self.image_size, self.image_size)
                
                print(f"      YOLO: [{yolo_bbox[0]:.4f}, {yolo_bbox[1]:.4f}, {yolo_bbox[2]:.4f}, {yolo_bbox[3]:.4f}]")
                
                # 최종 검증
                if all(0.0 <= coord <= 1.0 for coord in yolo_bbox):
                    cat_id_str = str(cat_id)
                    if cat_id_str in self.class_mapping:
                        class_id = self.class_mapping[cat_id_str]
                        class_name = categories[cat_id_str]['name']
                        yolo_labels.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")
                        print(f"      ✅ 유효: {class_name} (ID: {class_id})")
                        valid_boxes += 1
                else:
                    print(f"      ❌ YOLO 좌표 범위 초과")
                    invalid_boxes += 1
                    bbox_errors.append({
                        'image': filename,
                        'yolo_bbox': yolo_bbox,
                        'error': 'yolo_bounds'
                    })
            
            print(f"   📊 유효: {valid_boxes}, 무효: {invalid_boxes}")
            
            # 파일 저장
            output_image_path = self.target_path / 'images' / filename
            output_label_path = self.target_path / 'labels' / (Path(filename).stem + '.txt')
            
            cv2.imwrite(str(output_image_path), processed_image)
            
            if yolo_labels:
                with open(output_label_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))
            else:
                output_label_path.touch()
            
            processed_count += 1
        
        print(f"\n📊 처리 결과:")
        print(f"   성공: {processed_count}개")
        print(f"   실패: {error_count}개")
        print(f"   바운딩 박스 오류: {len(bbox_errors)}개")
        
        if bbox_errors:
            print(f"\n❌ 바운딩 박스 오류 샘플:")
            for i, error in enumerate(bbox_errors[:5]):
                print(f"   {i+1}. {error}")

if __name__ == "__main__":
    processor = SingleJSONProcessor()
    processor.process_single_json("/mnt/f/ds2_complete/deepscores-complete-0_test.json", max_images=5)