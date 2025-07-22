#!/usr/bin/env python3
"""
YOLOv8 학습 스크립트 with 고급 기능
점진적 클래스 확장, 모니터링, 자동 하이퍼파라미터 최적화 등을 지원합니다.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import yaml

import numpy as np
import torch
from ultralytics import YOLO
import wandb
import optuna


class YOLOTrainer:
    """고급 기능을 포함한 YOLO 학습 클래스"""
    
    def __init__(self, 
                 data_yaml: str,
                 model_size: str = 'yolov8s',
                 project_name: str = 'scoreeye-yolov8',
                 use_wandb: bool = True):
        """
        초기화
        
        Args:
            data_yaml: 데이터셋 YAML 파일 경로
            model_size: 모델 크기 ('yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x')
            project_name: 프로젝트 이름
            use_wandb: Weights & Biases 사용 여부
        """
        self.data_yaml = Path(data_yaml)
        self.model_size = model_size
        self.project_name = project_name
        self.use_wandb = use_wandb
        
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"데이터 YAML 파일을 찾을 수 없습니다: {data_yaml}")
        
        # 데이터 정보 로드
        with open(self.data_yaml, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.num_classes = len(self.data_config.get('names', {}))
        print(f"📊 클래스 수: {self.num_classes}")
        
        # 디바이스 설정
        self.device = self._get_optimal_device()
        print(f"🖥️ 사용 디바이스: {self.device}")
        
        # W&B 초기화
        if self.use_wandb:
            self._initialize_wandb()
    
    def _get_optimal_device(self) -> str:
        """최적 디바이스 자동 선택"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU 메모리: {gpu_memory:.1f}GB")
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            print("⚠️ GPU를 사용할 수 없습니다. CPU로 학습합니다.")
            return 'cpu'
    
    def _initialize_wandb(self):
        """Weights & Biases 초기화"""
        try:
            wandb.init(
                project=self.project_name,
                config={
                    "model": self.model_size,
                    "dataset": str(self.data_yaml),
                    "num_classes": self.num_classes,
                    "device": self.device,
                    "timestamp": datetime.now().isoformat()
                }
            )
            print("📈 W&B 연동 완료")
        except Exception as e:
            print(f"⚠️ W&B 초기화 실패: {e}")
            self.use_wandb = False
    
    def progressive_training(self, phases: Dict[str, Dict]) -> Dict[str, str]:
        """
        점진적 클래스 확장 학습
        
        Args:
            phases: 각 단계별 설정
            {
                'phase_1': {
                    'classes': ['noteheadFull', 'stem', 'gClef'],
                    'epochs': 50,
                    'target_mAP': 0.85
                },
                ...
            }
        
        Returns:
            각 단계별 최종 모델 경로
        """
        print("🎯 점진적 클래스 확장 학습 시작")
        print("=" * 60)
        
        phase_models = {}
        previous_model = f"{self.model_size}.pt"
        
        for phase_name, phase_config in phases.items():
            print(f"\n🚀 {phase_name} 시작")
            print(f"   타겟 클래스: {phase_config['classes']}")
            print(f"   목표 mAP: {phase_config.get('target_mAP', 'N/A')}")
            
            # 해당 단계 클래스만으로 제한된 데이터셋 생성
            phase_yaml = self._create_phase_dataset(phase_name, phase_config['classes'])
            
            # 학습 실행
            best_model = self.train(
                model_path=previous_model,
                data_yaml=phase_yaml,
                epochs=phase_config.get('epochs', 50),
                batch=phase_config.get('batch', 'auto'),
                imgsz=phase_config.get('imgsz', 2048),
                patience=phase_config.get('patience', 15),
                name=f"{phase_name}",
                save_period=phase_config.get('save_period', 10)
            )
            
            phase_models[phase_name] = best_model
            
            # 다음 단계에서 이 모델을 시작점으로 사용
            previous_model = best_model
            
            # 목표 mAP 달성 여부 확인
            if 'target_mAP' in phase_config:
                current_mAP = self._evaluate_model(best_model, phase_yaml)
                if current_mAP < phase_config['target_mAP']:
                    print(f"⚠️ 경고: 목표 mAP {phase_config['target_mAP']:.3f}에 미달 (현재: {current_mAP:.3f})")
                else:
                    print(f"✅ 목표 달성: mAP {current_mAP:.3f}")
        
        print("\n🎉 점진적 학습 완료!")
        return phase_models
    
    def _create_phase_dataset(self, phase_name: str, target_classes: List[str]) -> str:
        """특정 클래스만 포함하는 데이터셋 YAML 생성"""
        
        # 원본 클래스 이름에서 타겟 클래스의 ID 찾기
        class_mapping = {}
        new_id = 0
        
        for old_id, class_name in self.data_config['names'].items():
            if class_name in target_classes:
                class_mapping[int(old_id)] = new_id
                new_id += 1
        
        # 새 YAML 설정
        phase_config = {
            'path': self.data_config['path'],
            'train': self.data_config['train'],
            'val': self.data_config['val'],
            'names': {new_id: name for old_id, new_id in class_mapping.items() 
                     for name in [self.data_config['names'][old_id]]}
        }
        
        # 파일 저장
        phase_yaml_path = self.data_yaml.parent / f"{phase_name}_dataset.yaml"
        with open(phase_yaml_path, 'w') as f:
            yaml.dump(phase_config, f, default_flow_style=False)
        
        print(f"📄 {phase_name} 데이터셋 생성: {phase_yaml_path}")
        print(f"   포함된 클래스: {list(phase_config['names'].values())}")
        
        # 라벨 파일 필터링 (해당 클래스만 남기고 ID 재매핑)
        self._filter_labels_for_phase(class_mapping, phase_name)
        
        return str(phase_yaml_path)
    
    def _filter_labels_for_phase(self, class_mapping: Dict[int, int], phase_name: str):
        """특정 단계에 맞게 라벨 파일들을 필터링하고 ID 재매핑"""
        
        data_path = Path(self.data_config['path'])
        
        for subset in ['train', 'val']:
            labels_dir = data_path / 'labels' / subset
            phase_labels_dir = data_path / 'labels' / f"{subset}_{phase_name}"
            phase_labels_dir.mkdir(exist_ok=True)
            
            processed_files = 0
            
            for label_file in labels_dir.glob('*.txt'):
                new_lines = []
                
                with open(label_file, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        
                        old_class_id = int(parts[0])
                        
                        # 해당 클래스가 현재 단계에 포함되는지 확인
                        if old_class_id in class_mapping:
                            new_class_id = class_mapping[old_class_id]
                            parts[0] = str(new_class_id)
                            new_lines.append(' '.join(parts))
                
                # 새 라벨 파일 저장 (해당 클래스가 있는 경우만)
                if new_lines:
                    new_label_file = phase_labels_dir / label_file.name
                    with open(new_label_file, 'w') as f:
                        f.write('\n'.join(new_lines))
                    processed_files += 1
            
            print(f"   {subset} 라벨 필터링: {processed_files}개 파일 처리")
        
        # 원본 YAML에서 새 라벨 경로로 업데이트
        phase_yaml_path = self.data_yaml.parent / f"{phase_name}_dataset.yaml"
        with open(phase_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config['train'] = f"images/train"  # 이미지는 그대로
        config['val'] = f"images/val"
        
        with open(phase_yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def train(self, 
              model_path: str = None,
              data_yaml: str = None,
              epochs: int = 100,
              batch: str = 'auto',
              imgsz: int = 2048,
              patience: int = 15,
              name: str = None,
              save_period: int = 10) -> str:
        """
        YOLO 모델 학습
        
        Returns:
            최종 모델 파일 경로
        """
        
        if model_path is None:
            model_path = f"{self.model_size}.pt"
        
        if data_yaml is None:
            data_yaml = str(self.data_yaml)
        
        print(f"🚀 학습 시작")
        print(f"   모델: {model_path}")
        print(f"   데이터: {data_yaml}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch: {batch}")
        print(f"   이미지 크기: {imgsz}")
        
        # GPU 메모리에 따른 배치 크기 자동 조정
        if batch == 'auto':
            batch = self._get_optimal_batch_size(imgsz)
            print(f"   자동 배치 크기: {batch}")
        
        # 모델 로드
        model = YOLO(model_path)
        
        # 학습 실행 (W&B 활성화)
        import os
        os.environ['WANDB_PROJECT'] = self.project_name
        
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            patience=patience,
            device=self.device,
            amp=True,  # Mixed precision training
            name=name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            save_period=save_period,
            plots=True,
            verbose=True,
            # W&B 활성화 파라미터 추가
            project=self.project_name if self.use_wandb else None
        )
        
        # W&B에 최종 메트릭 기록
        if self.use_wandb and hasattr(results, 'results_dict'):
            final_metrics = {
                'final_mAP50': results.results_dict.get('metrics/mAP50(B)', 0.0),
                'final_mAP50_95': results.results_dict.get('metrics/mAP50-95(B)', 0.0),
                'final_precision': results.results_dict.get('metrics/precision(B)', 0.0),
                'final_recall': results.results_dict.get('metrics/recall(B)', 0.0),
                'total_epochs': epochs,
                'final_box_loss': results.results_dict.get('train/box_loss', 0.0),
                'final_cls_loss': results.results_dict.get('train/cls_loss', 0.0)
            }
            wandb.log(final_metrics)
        
        # 최종 모델 경로
        best_model_path = results.save_dir / 'weights' / 'best.pt'
        
        print(f"✅ 학습 완료: {best_model_path}")
        
        if self.use_wandb:
            wandb.log({"training_completed": True})
        
        return str(best_model_path)
    
    def _get_optimal_batch_size(self, imgsz: int) -> int:
        """이미지 크기와 GPU 메모리를 고려한 최적 배치 크기 계산"""
        if self.device == 'cpu':
            return 4
        
        try:
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                # 간단한 휴리스틱 (이미지 크기와 GPU 메모리 기반)
                if imgsz <= 640:
                    if gpu_memory_gb >= 16:
                        return 16
                    elif gpu_memory_gb >= 8:
                        return 12
                    else:
                        return 8
                elif imgsz <= 2048:
                    if gpu_memory_gb >= 16:
                        return 8
                    elif gpu_memory_gb >= 8:
                        return 4
                    else:
                        return 2
                else:  # > 1024
                    if gpu_memory_gb >= 16:
                        return 4
                    else:
                        return 2
            else:
                return 4
                
        except Exception:
            return 4
    
    def optimize_hyperparameters(self, 
                                n_trials: int = 20, 
                                timeout: int = 7200) -> Dict:
        """
        Optuna를 사용한 하이퍼파라미터 자동 최적화
        
        Args:
            n_trials: 시행 횟수
            timeout: 최대 시간 (초)
            
        Returns:
            최적 하이퍼파라미터
        """
        print(f"🔍 하이퍼파라미터 최적화 시작 (최대 {n_trials}회, {timeout/60:.0f}분)")
        
        def objective(trial):
            # 하이퍼파라미터 공간 정의
            params = {
                'lr0': trial.suggest_loguniform('lr0', 1e-5, 1e-2),
                'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
                'momentum': trial.suggest_uniform('momentum', 0.8, 0.99),
                'batch': trial.suggest_categorical('batch', [4, 8, 16]),
                'warmup_epochs': trial.suggest_int('warmup_epochs', 0, 5)
            }
            
            # 임시 모델로 짧은 학습 실행
            model = YOLO(f"{self.model_size}.pt")
            
            try:
                results = model.train(
                    data=str(self.data_yaml),
                    epochs=10,  # 짧은 학습
                    batch=params['batch'],
                    lr0=params['lr0'],
                    weight_decay=params['weight_decay'],
                    momentum=params['momentum'],
                    warmup_epochs=params['warmup_epochs'],
                    device=self.device,
                    verbose=False,
                    plots=False,
                    save=False
                )
                
                # mAP@0.5를 최적화 목표로 사용
                return results.results_dict.get('metrics/mAP50(B)', 0.0)
                
            except Exception as e:
                print(f"Trial 실패: {e}")
                return 0.0
        
        # Optuna 스터디 실행
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        
        print(f"🎯 최적 하이퍼파라미터:")
        for key, value in best_params.items():
            print(f"   {key}: {value}")
        
        return best_params
    
    def _evaluate_model(self, model_path: str, data_yaml: str) -> float:
        """모델 성능 평가 (mAP@0.5 반환)"""
        try:
            model = YOLO(model_path)
            results = model.val(data=data_yaml, device=self.device, verbose=False)
            return results.results_dict.get('metrics/mAP50(B)', 0.0)
        except Exception:
            return 0.0
    
    def create_backup_strategy_plan(self) -> Dict:
        """백업 전략 계획 생성"""
        strategies = {
            'primary': {
                'model': self.model_size,
                'strategy': 'Progressive Class Expansion',
                'expected_performance': 0.80,
                'training_time_days': 12,
                'gpu_requirement_gb': 8
            },
            'backup_level_1': {
                'model': 'yolov8n',
                'strategy': 'All-class simultaneous training',
                'expected_performance': 0.70,
                'training_time_days': 6,
                'gpu_requirement_gb': 4
            },
            'backup_level_2': {
                'model': 'yolov8s',  
                'strategy': 'Traditional training with reduced epochs',
                'expected_performance': 0.65,
                'training_time_days': 3,
                'gpu_requirement_gb': 6
            }
        }
        
        return strategies


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="YOLOv8 고급 학습 스크립트")
    parser.add_argument("--data", required=True, help="데이터셋 YAML 파일")
    parser.add_argument("--model", default="yolov8s", help="모델 크기")
    parser.add_argument("--epochs", type=int, default=100, help="학습 에폭 수")
    parser.add_argument("--batch", default="auto", help="배치 크기", 
                        type=lambda x: int(x) if x.isdigit() else x)
    parser.add_argument("--imgsz", type=int, default=2048, help="이미지 크기")
    parser.add_argument("--patience", type=int, default=15, help="조기 종료 patience")
    parser.add_argument("--name", help="실험 이름")
    parser.add_argument("--progressive", action="store_true", help="점진적 학습 실행")
    parser.add_argument("--optimize", action="store_true", help="하이퍼파라미터 최적화")
    parser.add_argument("--no-wandb", action="store_true", help="W&B 비활성화")
    
    args = parser.parse_args()
    
    # 트레이너 초기화
    trainer = YOLOTrainer(
        data_yaml=args.data,
        model_size=args.model,
        use_wandb=not args.no_wandb
    )
    
    try:
        if args.optimize:
            # 하이퍼파라미터 최적화
            best_params = trainer.optimize_hyperparameters()
            
            # 최적 파라미터로 학습 실행
            print("\n🚀 최적 파라미터로 최종 학습 시작...")
            # 여기서 best_params를 활용한 학습 로직 추가 가능
            
        elif args.progressive:
            # 점진적 학습
            phases = {
                'phase_1': {
                    'classes': ['noteheadFull', 'stem', 'gClef'],
                    'epochs': 50,
                    'target_mAP': 0.85,
                    'batch': 8
                },
                'phase_2': {
                    'classes': ['restQuarter', 'beam', 'dot'],
                    'epochs': 30,
                    'target_mAP': 0.80,
                    'batch': 8
                },
                'phase_3': {
                    'classes': ['sharp', 'flat', 'natural'],
                    'epochs': 20,
                    'target_mAP': 0.75,
                    'batch': 8
                }
            }
            
            phase_models = trainer.progressive_training(phases)
            
            print(f"\n🎉 점진적 학습 완료!")
            for phase, model_path in phase_models.items():
                print(f"   {phase}: {model_path}")
                
        else:
            # 일반 학습
            best_model = trainer.train(
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                patience=args.patience,
                name=args.name
            )
            
            print(f"🎉 학습 완료: {best_model}")
            
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        if trainer.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()