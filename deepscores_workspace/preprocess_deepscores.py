#!/usr/bin/env python3
"""
DeepScores 데이터셋 전처리 및 품질 검증 스크립트
NPZ 어노테이션을 YOLO 형식으로 변환하고 데이터 품질을 검증합니다.
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DeepScoresPreprocessor:
    """DeepScores 데이터셋 전처리 클래스"""
    
    def __init__(self, raw_data_dir: str, output_dir: str, pilot_mode: bool = False, sample_size: int = 1000):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.pilot_mode = pilot_mode
        self.sample_size = sample_size
        
        # 목표 클래스 정의 (점진적 확장을 위해 Phase별로 분류)
        self.target_classes_phases = {
            'phase_1': ['noteheadFull', 'stem', 'gClef'],
            'phase_2': ['restQuarter', 'beam', 'dot'],
            'phase_3': ['sharp', 'flat', 'natural'],
            'phase_4': ['timeSig4_4', 'keySigFlat1', 'noteheadHalf']
        }
        
        # 전체 목표 클래스 (초기에는 Phase 1-2만 사용)
        self.target_classes = {}
        class_id = 0
        for phase_classes in self.target_classes_phases.values():
            for class_name in phase_classes:
                self.target_classes[class_name] = class_id
                class_id += 1
        
        self.image_size = 1024  # DeepScores 고해상도 처리
        self.train_ratio = 0.85
        
        # 검증 결과 저장
        self.validation_results = {}
        
    def convert_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """DeepScores bbox [x_min, y_min, x_max, y_max] -> YOLO 포맷 변환"""
        dw = 1.0 / img_width
        dh = 1.0 / img_height
        
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_center = (bbox[1] + bbox[3]) / 2.0
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        return (x_center * dw, y_center * dh, width * dw, height * dh)
    
    def validate_bbox(self, bbox: Tuple[float, float, float, float]) -> bool:
        """YOLO 바운딩 박스 유효성 검사"""
        x_center, y_center, width, height = bbox
        
        # 0-1 범위 확인
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            return False
        if not (0 < width <= 1 and 0 < height <= 1):
            return False
            
        # 경계 확인
        if (x_center - width/2 < 0) or (x_center + width/2 > 1):
            return False
        if (y_center - height/2 < 0) or (y_center + height/2 > 1):
            return False
            
        return True
    
    def smart_dataset_sampling(self, all_images: List[str]) -> List[str]:
        """클래스 균형을 고려한 지능적 데이터 샘플링"""
        if not self.pilot_mode:
            return all_images
        
        print(f"🎯 Pilot Mode: {self.sample_size}개 샘플 선택 중...")
        
        # 각 이미지의 클래스 분포 분석
        image_class_counts = {}
        target_class_names = set(self.target_classes.keys())
        
        for img_name in tqdm(all_images[:5000], desc="클래스 분포 분석"):  # 최대 5000개만 분석
            npz_path = self.raw_data_dir / "annotations" / (img_name.replace('.png', '.npz'))
            if not npz_path.exists():
                continue
                
            try:
                annotations = np.load(npz_path, allow_pickle=True)['arr_0']
                class_counts = Counter()
                
                for ann in annotations:
                    if hasattr(ann, 'item'):
                        ann = ann.item()
                    class_name = ann.get('class_name', '')
                    
                    if class_name in target_class_names:
                        class_counts[class_name] += 1
                
                image_class_counts[img_name] = class_counts
                
            except Exception as e:
                print(f"Warning: 파일 {npz_path} 분석 실패: {e}")
                continue
        
        # 균형잡힌 샘플링
        sampled_images = []
        class_samples = defaultdict(list)
        
        # 클래스별로 이미지 그룹핑
        for img_name, class_counts in image_class_counts.items():
            for class_name in class_counts.keys():
                class_samples[class_name].append((img_name, class_counts[class_name]))
        
        # 각 클래스에서 균등하게 샘플링
        samples_per_class = max(50, self.sample_size // len(self.target_classes))  # 클래스당 최소 50개
        
        for class_name, images in class_samples.items():
            # 해당 클래스가 많이 포함된 이미지 우선 선택
            images.sort(key=lambda x: x[1], reverse=True)
            selected = [img[0] for img in images[:samples_per_class]]
            sampled_images.extend(selected)
        
        # 중복 제거 및 최종 샘플 수 조정
        sampled_images = list(set(sampled_images))
        
        if len(sampled_images) > self.sample_size:
            sampled_images = random.sample(sampled_images, self.sample_size)
        
        print(f"✅ {len(sampled_images)}개 샘플 선택 완료")
        return sampled_images
    
    def process_dataset(self):
        """메인 데이터셋 처리 함수"""
        print("🚀 DeepScores 데이터셋 전처리 시작")
        print(f"원본 데이터: {self.raw_data_dir}")
        print(f"출력 디렉터리: {self.output_dir}")
        print(f"Pilot Mode: {self.pilot_mode} (샘플 크기: {self.sample_size})")
        print("=" * 60)
        
        # 출력 디렉터리 생성
        for subset in ['train', 'val']:
            (self.output_dir / 'images' / subset).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / subset).mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 목록 가져오기
        images_dir = self.raw_data_dir / "images_png"
        if not images_dir.exists():
            images_dir = self.raw_data_dir / "images"  # 대체 경로
            
        if not images_dir.exists():
            raise FileNotFoundError(f"이미지 디렉터리를 찾을 수 없습니다: {images_dir}")
        
        all_images = [f.name for f in images_dir.glob('*.png')]
        print(f"📁 총 이미지 수: {len(all_images)}")
        
        if len(all_images) == 0:
            raise ValueError("처리할 이미지가 없습니다.")
        
        # 스마트 샘플링
        selected_images = self.smart_dataset_sampling(all_images)
        
        # Train/Validation 분할
        train_images, val_images = train_test_split(
            selected_images, 
            train_size=self.train_ratio, 
            random_state=42
        )
        
        print(f"📊 Train: {len(train_images)}, Validation: {len(val_images)}")
        
        # 데이터셋 처리
        processing_stats = {
            'total_images': 0,
            'processed_images': 0,
            'total_annotations': 0,
            'valid_annotations': 0,
            'invalid_bboxes': 0,
            'missing_files': 0,
            'class_distribution': Counter()
        }
        
        for subset, image_list in [('train', train_images), ('val', val_images)]:
            print(f"\n🔄 {subset.upper()} 데이터셋 처리 중...")
            
            img_out_path = self.output_dir / 'images' / subset
            lbl_out_path = self.output_dir / 'labels' / subset
            
            for img_name in tqdm(image_list, desc=f"Processing {subset}"):
                processing_stats['total_images'] += 1
                
                try:
                    # 원본 이미지 처리
                    img_path = images_dir / img_name
                    if not img_path.exists():
                        processing_stats['missing_files'] += 1
                        continue
                    
                    original_img = cv2.imread(str(img_path))
                    if original_img is None:
                        processing_stats['missing_files'] += 1
                        continue
                        
                    h, w, _ = original_img.shape
                    
                    # 이미지 리사이즈 및 저장
                    resized_img = cv2.resize(original_img, (self.image_size, self.image_size))
                    cv2.imwrite(str(img_out_path / img_name), resized_img)
                    
                    # NPZ 어노테이션 로드
                    npz_path = self.raw_data_dir / "annotations" / img_name.replace('.png', '.npz')
                    if not npz_path.exists():
                        # 빈 라벨 파일 생성
                        (lbl_out_path / img_name.replace('.png', '.txt')).touch()
                        processing_stats['missing_files'] += 1
                        continue
                    
                    annotations = np.load(npz_path, allow_pickle=True)['arr_0']
                    
                    yolo_labels = []
                    for ann in annotations:
                        processing_stats['total_annotations'] += 1
                        
                        if hasattr(ann, 'item'):
                            ann = ann.item()
                        
                        class_name = ann.get('class_name', '')
                        if class_name not in self.target_classes:
                            continue
                        
                        class_id = self.target_classes[class_name]
                        bbox = ann.get('bbox', [])
                        
                        if len(bbox) != 4:
                            processing_stats['invalid_bboxes'] += 1
                            continue
                        
                        # YOLO 포맷으로 변환
                        yolo_bbox = self.convert_to_yolo(bbox, w, h)
                        
                        # 바운딩 박스 유효성 검사
                        if not self.validate_bbox(yolo_bbox):
                            processing_stats['invalid_bboxes'] += 1
                            continue
                        
                        yolo_labels.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")
                        processing_stats['valid_annotations'] += 1
                        processing_stats['class_distribution'][class_name] += 1
                    
                    # 라벨 파일 저장
                    label_file = lbl_out_path / img_name.replace('.png', '.txt')
                    with open(label_file, 'w') as f:
                        f.write('\n'.join(yolo_labels))
                    
                    processing_stats['processed_images'] += 1
                    
                except Exception as e:
                    print(f"오류: {img_name} 처리 실패 - {str(e)}")
                    continue
        
        # 처리 결과 요약
        self._print_processing_summary(processing_stats)
        
        # YAML 설정 파일 생성
        self._create_dataset_yaml()
        
        # 데이터 품질 검증 실행
        self._validate_dataset_quality(processing_stats)
        
        print("\n🎉 데이터셋 전처리 완료!")
    
    def _print_processing_summary(self, stats: Dict):
        """처리 결과 요약 출력"""
        print("\n" + "=" * 60)
        print("📈 처리 결과 요약")
        print("=" * 60)
        print(f"총 이미지 수: {stats['total_images']}")
        print(f"성공적으로 처리된 이미지: {stats['processed_images']}")
        print(f"총 어노테이션 수: {stats['total_annotations']}")
        print(f"유효한 어노테이션 수: {stats['valid_annotations']}")
        print(f"무효한 바운딩 박스: {stats['invalid_bboxes']}")
        print(f"누락된 파일: {stats['missing_files']}")
        
        print(f"\n📊 클래스 분포:")
        for class_name, count in stats['class_distribution'].most_common():
            percentage = (count / stats['valid_annotations']) * 100 if stats['valid_annotations'] > 0 else 0
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    def _create_dataset_yaml(self):
        """YAML 설정 파일 생성"""
        yaml_content = f"""# DeepScores Dataset Configuration for YOLOv8
path: {self.output_dir.absolute()}
train: images/train
val: images/val

# Classes ({len(self.target_classes)} classes)
names:
"""
        
        for class_name, class_id in sorted(self.target_classes.items(), key=lambda x: x[1]):
            yaml_content += f"  {class_id}: {class_name}\n"
        
        yaml_file = self.output_dir / "deepscores.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        print(f"📄 YAML 설정 파일 생성: {yaml_file}")
    
    def _validate_dataset_quality(self, processing_stats: Dict):
        """데이터셋 품질 검증"""
        print("\n🔍 데이터셋 품질 검증 중...")
        
        validation_results = {
            'processing_stats': processing_stats,
            'image_label_pairs': self._check_image_label_pairs(),
            'class_imbalance': self._analyze_class_imbalance(processing_stats['class_distribution']),
            'sample_visualization': self._create_sample_visualization()
        }
        
        # 검증 리포트 저장
        report_dir = Path("validation_reports")
        report_dir.mkdir(exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = report_dir / f"preprocessing_validation_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            # Counter 객체를 dict로 변환하여 JSON 직렬화 가능하게 함
            serializable_results = validation_results.copy()
            serializable_results['processing_stats']['class_distribution'] = dict(processing_stats['class_distribution'])
            json.dump(serializable_results, f, indent=2)
        
        print(f"📋 검증 리포트 저장: {report_file}")
        
        self.validation_results = validation_results
    
    def _check_image_label_pairs(self) -> Dict:
        """이미지-라벨 쌍 매칭 확인"""
        results = {'train': {}, 'val': {}}
        
        for subset in ['train', 'val']:
            img_dir = self.output_dir / 'images' / subset
            lbl_dir = self.output_dir / 'labels' / subset
            
            img_files = set(f.stem for f in img_dir.glob('*.png'))
            lbl_files = set(f.stem for f in lbl_dir.glob('*.txt'))
            
            results[subset] = {
                'total_images': len(img_files),
                'total_labels': len(lbl_files),
                'missing_labels': len(img_files - lbl_files),
                'orphan_labels': len(lbl_files - img_files),
                'matched_pairs': len(img_files & lbl_files)
            }
        
        return results
    
    def _analyze_class_imbalance(self, class_distribution: Counter) -> Dict:
        """클래스 불균형 분석"""
        if not class_distribution:
            return {'status': 'no_data'}
        
        total = sum(class_distribution.values())
        class_ratios = {cls: count/total for cls, count in class_distribution.items()}
        
        # 불균형 심각도 계산
        max_ratio = max(class_ratios.values())
        min_ratio = min(class_ratios.values())
        imbalance_ratio = max_ratio / min_ratio if min_ratio > 0 else float('inf')
        
        # 권장사항 생성
        recommendations = []
        if imbalance_ratio > 50:
            recommendations.append("심각한 클래스 불균형 - weighted loss 함수 사용 권장")
        if imbalance_ratio > 20:
            recommendations.append("클래스별 데이터 augmentation 차등 적용 권장")
        if min_ratio < 0.01:  # 1% 미만
            recommendations.append("희귀 클래스에 대한 oversampling 권장")
        
        return {
            'imbalance_ratio': imbalance_ratio,
            'class_ratios': class_ratios,
            'recommendations': recommendations
        }
    
    def _create_sample_visualization(self, num_samples: int = 10) -> Dict:
        """샘플 이미지 시각화"""
        print("📸 샘플 시각화 생성 중...")
        
        try:
            train_img_dir = self.output_dir / 'images' / 'train'
            train_lbl_dir = self.output_dir / 'labels' / 'train'
            
            img_files = list(train_img_dir.glob('*.png'))
            if len(img_files) == 0:
                return {'status': 'no_images'}
            
            sample_files = random.sample(img_files, min(num_samples, len(img_files)))
            
            # 시각화 생성
            fig, axes = plt.subplots(2, 5, figsize=(20, 8))
            axes = axes.flatten()
            
            visualized_count = 0
            
            for i, img_file in enumerate(sample_files):
                if i >= len(axes):
                    break
                
                # 이미지 로드
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                    
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 라벨 로드
                label_file = train_lbl_dir / (img_file.stem + '.txt')
                
                ax = axes[i]
                ax.imshow(image_rgb)
                ax.set_title(f"{img_file.stem}")
                ax.axis('off')
                
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        lines = f.read().strip().split('\n')
                        
                    h, w = image_rgb.shape[:2]
                    
                    for line in lines:
                        if not line.strip():
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            continue
                        
                        class_id, x_center, y_center, width, height = map(float, parts)
                        
                        # YOLO -> 픽셀 좌표 변환
                        x1 = (x_center - width/2) * w
                        y1 = (y_center - height/2) * h
                        box_width = width * w
                        box_height = height * h
                        
                        rect = patches.Rectangle(
                            (x1, y1), box_width, box_height,
                            linewidth=2, edgecolor='red', facecolor='none'
                        )
                        ax.add_patch(rect)
                        
                        # 클래스 이름 표시
                        class_name = [name for name, id in self.target_classes.items() if id == int(class_id)]
                        class_name = class_name[0] if class_name else f"Class_{int(class_id)}"
                        
                        ax.text(x1, y1-5, class_name, color='red', fontsize=8, 
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
                
                visualized_count += 1
            
            # 빈 subplot 숨기기
            for i in range(visualized_count, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            # 시각화 저장
            vis_dir = Path("validation_reports")
            vis_dir.mkdir(exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            vis_file = vis_dir / f"sample_visualization_{timestamp}.png"
            
            plt.savefig(vis_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            return {
                'status': 'success',
                'visualized_samples': visualized_count,
                'output_file': str(vis_file)
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="DeepScores 데이터셋 전처리")
    parser.add_argument("--raw-data", default="raw_data", help="원본 DeepScores 데이터 디렉터리")
    parser.add_argument("--output", default="data", help="출력 디렉터리")
    parser.add_argument("--pilot-mode", action="store_true", help="Pilot 모드 (샘플 데이터만 처리)")
    parser.add_argument("--sample-size", type=int, default=1000, help="Pilot 모드 샘플 크기")
    
    args = parser.parse_args()
    
    processor = DeepScoresPreprocessor(
        raw_data_dir=args.raw_data,
        output_dir=args.output,
        pilot_mode=args.pilot_mode,
        sample_size=args.sample_size
    )
    
    try:
        processor.process_dataset()
        print("✅ 전처리 성공적으로 완료!")
        
    except Exception as e:
        print(f"❌ 전처리 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()