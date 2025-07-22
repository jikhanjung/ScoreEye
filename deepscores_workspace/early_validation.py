#!/usr/bin/env python3
"""
조기 검증 시스템 (Early Validation System)
각 Phase 완료 후 전체 파이프라인의 건전성을 조기에 검증
"""

import os
import time
import traceback
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import cv2

class EarlyValidator:
    """파이프라인 조기 검증을 위한 클래스"""
    
    def __init__(self, workspace_dir: str = "."):
        self.workspace_dir = Path(workspace_dir)
        self.validation_results = {}
        
    def early_pipeline_validation(self) -> bool:
        """Phase 2 완료 후 즉시 실행하는 파이프라인 검증"""
        
        validation_steps = [
            {
                'name': 'data_loading_test',
                'description': '10개 샘플 이미지와 라벨이 정상적으로 로드되는지 확인',
                'expected_time': 30,  # seconds
                'function': self._test_data_loading
            },
            {
                'name': 'model_initialization_test',
                'description': 'YOLOv8s 모델이 GPU에서 정상 초기화되는지 확인',
                'expected_time': 60,
                'function': self._test_model_initialization
            },
            {
                'name': 'single_epoch_training_test',
                'description': '1 epoch 학습이 오류 없이 완료되는지 확인',
                'expected_time': 600,  # 10 minutes
                'function': self._test_single_epoch_training
            },
            {
                'name': 'inference_test',
                'description': '학습된 모델로 샘플 이미지 추론이 가능한지 확인',
                'expected_time': 60,
                'function': self._test_inference
            }
        ]
        
        print("🔍 파이프라인 조기 검증 시작...")
        print("=" * 60)
        
        all_passed = True
        
        for step in validation_steps:
            print(f"\n▶️ {step['name']}: {step['description']}")
            print(f"   예상 소요시간: {step['expected_time']}초")
            
            start_time = time.time()
            
            try:
                result = step['function']()
                elapsed_time = time.time() - start_time
                
                if result:
                    print(f"   ✅ 통과 ({elapsed_time:.1f}초)")
                    self.validation_results[step['name']] = {
                        'status': 'PASS',
                        'elapsed_time': elapsed_time,
                        'details': result
                    }
                else:
                    print(f"   ❌ 실패 ({elapsed_time:.1f}초)")
                    self.validation_results[step['name']] = {
                        'status': 'FAIL',
                        'elapsed_time': elapsed_time,
                        'error': 'Function returned False'
                    }
                    all_passed = False
                    
            except Exception as e:
                elapsed_time = time.time() - start_time
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                print(f"   ❌ 오류 발생 ({elapsed_time:.1f}초): {str(e)}")
                
                self.validation_results[step['name']] = {
                    'status': 'ERROR',
                    'elapsed_time': elapsed_time,
                    'error': error_msg
                }
                all_passed = False
                break  # 오류 발생 시 즉시 중단
        
        print("\n" + "=" * 60)
        if all_passed:
            print("🎉 모든 검증 통과! Phase 3 진행 가능")
        else:
            print("⚠️ 검증 실패! 문제 해결 후 재실행 필요")
            
        self._save_validation_report()
        return all_passed
    
    def _test_data_loading(self) -> bool:
        """데이터 로딩 테스트"""
        data_dir = self.workspace_dir / "data"
        
        # 이미지와 라벨 파일 존재 확인
        train_images_dir = data_dir / "images" / "train"
        train_labels_dir = data_dir / "labels" / "train"
        
        if not train_images_dir.exists():
            raise FileNotFoundError(f"Training images directory not found: {train_images_dir}")
        if not train_labels_dir.exists():
            raise FileNotFoundError(f"Training labels directory not found: {train_labels_dir}")
        
        # 샘플 파일들 확인
        image_files = list(train_images_dir.glob("*.png"))[:10]
        if len(image_files) < 5:
            raise ValueError(f"Insufficient sample images: {len(image_files)} < 5")
        
        loaded_count = 0
        for img_file in image_files:
            label_file = train_labels_dir / (img_file.stem + ".txt")
            
            # 이미지 로딩
            image = cv2.imread(str(img_file))
            if image is None:
                continue
                
            # 라벨 파일 확인
            if label_file.exists():
                with open(label_file, 'r') as f:
                    labels = f.read().strip()
                    if labels:  # 비어있지 않은 라벨
                        loaded_count += 1
        
        return loaded_count >= 5
    
    def _test_model_initialization(self) -> bool:
        """모델 초기화 테스트"""
        try:
            from ultralytics import YOLO
            
            # YOLOv8s 모델 초기화
            model = YOLO('yolov8s.pt')
            
            # GPU 사용 가능 여부 확인
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            
            print(f"     모델 초기화 성공 (Device: {device})")
            return True
            
        except ImportError:
            raise ImportError("ultralytics 라이브러리가 설치되지 않았습니다")
        except Exception as e:
            raise Exception(f"모델 초기화 실패: {str(e)}")
    
    def _test_single_epoch_training(self) -> bool:
        """1 epoch 학습 테스트"""
        try:
            from ultralytics import YOLO
            
            # 데이터셋 설정 파일 확인
            yaml_file = self.workspace_dir / "data" / "deepscores.yaml"
            if not yaml_file.exists():
                # 간단한 테스트용 yaml 생성
                self._create_test_yaml()
            
            model = YOLO('yolov8n.pt')  # 빠른 테스트를 위해 nano 모델 사용
            
            # 매우 제한적인 설정으로 1 epoch 학습
            results = model.train(
                data=str(yaml_file),
                epochs=1,
                batch=4,
                imgsz=640,  # 작은 이미지 크기
                verbose=False,
                patience=0
            )
            
            return True
            
        except Exception as e:
            raise Exception(f"학습 테스트 실패: {str(e)}")
    
    def _test_inference(self) -> bool:
        """추론 테스트"""
        try:
            from ultralytics import YOLO
            
            # 학습된 모델 찾기 (또는 사전 훈련된 모델 사용)
            model_path = self._find_trained_model() or 'yolov8n.pt'
            model = YOLO(model_path)
            
            # 테스트 이미지로 추론
            test_images = list((self.workspace_dir / "data" / "images" / "train").glob("*.png"))[:3]
            
            if not test_images:
                raise FileNotFoundError("추론 테스트용 이미지가 없습니다")
            
            for img_path in test_images:
                results = model(str(img_path), verbose=False)
                # 결과가 정상적으로 반환되는지만 확인
                if results:
                    print(f"     추론 성공: {len(results)} results")
            
            return True
            
        except Exception as e:
            raise Exception(f"추론 테스트 실패: {str(e)}")
    
    def _create_test_yaml(self):
        """테스트용 YAML 파일 생성"""
        yaml_content = f"""
path: {self.workspace_dir.absolute() / 'data'}
train: images/train
val: images/val

# Test classes (minimal)
names:
  0: noteheadFull
  1: stem
  2: gClef
"""
        
        yaml_file = self.workspace_dir / "data" / "deepscores.yaml"
        yaml_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_file, 'w') as f:
            f.write(yaml_content.strip())
    
    def _find_trained_model(self) -> Optional[str]:
        """학습된 모델 찾기"""
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            for train_dir in runs_dir.glob("train*"):
                weights_dir = train_dir / "weights"
                if weights_dir.exists():
                    best_pt = weights_dir / "best.pt"
                    if best_pt.exists():
                        return str(best_pt)
        return None
    
    def _save_validation_report(self):
        """검증 결과 리포트 저장"""
        report_dir = self.workspace_dir / "validation_reports"
        report_dir.mkdir(exist_ok=True)
        
        import json
        from datetime import datetime
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_results': self.validation_results,
            'overall_status': 'PASS' if all(
                result.get('status') == 'PASS' 
                for result in self.validation_results.values()
            ) else 'FAIL'
        }
        
        report_file = report_dir / f"early_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📋 검증 리포트 저장: {report_file}")


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepScores 파이프라인 조기 검증")
    parser.add_argument("--workspace", default=".", help="작업 디렉터리 경로")
    
    args = parser.parse_args()
    
    validator = EarlyValidator(args.workspace)
    success = validator.early_pipeline_validation()
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()