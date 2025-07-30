#!/usr/bin/env python3
"""
Stage 3 YOLOv8 훈련 스크립트 
작성일: 2025-07-30
목적: Stage 2 오류 수정 후 YOLOv8m으로 고성능 훈련
개선사항:
- YOLOv8m 사용 (더 높은 정확도)
- clefG 포함된 올바른 클래스로 훈련
- 안정적인 메모리 관리 (배치 1)
- mosaic 0으로 메모리 절약
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO
import torch
import wandb
import yaml
from datetime import datetime

def setup_wandb(config):
    """Weights & Biases 설정"""
    wandb.init(
        project="deepscores-stage3",
        name=f"yolov8m-stage3-{config['image_size']}px-{datetime.now().strftime('%m%d_%H%M')}",
        config={
            'model': config['model_size'],
            'classes': config['classes'],
            'image_size': config['image_size'],
            'batch_size': config['batch_size'],
            'epochs': config['epochs'],
            'device': config['device'],
            'optimizer': config['optimizer'],
            'lr0': config['lr0'],
            'stage': 'stage3',
            'improvements': [
                'YOLOv8m 사용 (성능 향상)',
                'clefG 포함된 올바른 클래스',
                'beam/tie/slur 정확히 제외',
                '안정적 메모리 관리'
            ]
        }
    )

def calculate_class_weights(dataset_path):
    """클래스별 가중치 계산 (불균형 해결)"""
    print("📊 클래스 불균형 분석 중...")
    
    # stage1_classes.json에서 클래스 빈도 정보 로드
    if Path('stage1_classes.json').exists():
        with open('stage1_classes.json', 'r') as f:
            class_info = json.load(f)
    else:
        print("⚠️  stage1_classes.json 파일이 없습니다. 가중치 계산 스킵")
        return None
    
    # excluded_classes_info.json에서 제외된 클래스 정보 로드
    excluded_info_path = dataset_path / 'excluded_classes_info.json'
    if excluded_info_path.exists():
        with open(excluded_info_path, 'r') as f:
            excluded_info = json.load(f)
        excluded_classes = set(excluded_info['excluded_classes'])
    else:
        excluded_classes = {'42', '2', '122', '123', '121'}  # Stage 3 기본값
    
    # 제외되지 않은 클래스들만 필터링
    valid_classes = []
    for cls in class_info['classes']:
        if str(cls['id']) not in excluded_classes:
            valid_classes.append(cls)
    
    # 클래스별 빈도
    class_counts = {}
    for cls in valid_classes:
        class_counts[cls['id']] = cls['count']
    
    # 가중치 계산 (역빈도)
    total_samples = sum(class_counts.values())
    class_weights = {}
    
    for i, cls in enumerate(valid_classes):
        original_id = cls['id']
        frequency = cls['count'] / total_samples
        # 가중치 = 1 / sqrt(frequency) (너무 극단적이지 않게)
        weight = 1.0 / (frequency ** 0.5)
        class_weights[i] = weight
    
    print(f"✅ 클래스 가중치 계산 완료 ({len(valid_classes)}개 클래스)")
    return class_weights

def train_stage3(args):
    """Stage 3 모델 훈련"""
    print("🚀 Stage 3 YOLOv8m 훈련 시작")
    print(f"📅 작성일: 2025-07-30")
    print(f"🎯 목표: clefG 포함, YOLOv8m으로 고성능 음악 기호 검출")
    
    # 설정 로드
    config_file = 'stage3_preprocess_config.json'
    if Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    else:
        print(f"❌ 설정 파일을 찾을 수 없습니다: {config_file}")
        return
    
    # GPU/디바이스 설정
    if args.device != 'cpu':
        device = f'cuda:{args.device}'
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(int(args.device)).total_memory / 1024**3
            print(f"🎮 GPU 메모리: {gpu_memory:.1f}GB")
        else:
            print("⚠️  CUDA 사용 불가, CPU 모드로 전환")
            device = 'cpu'
    else:
        device = 'cpu'
        print("⚠️  CPU 모드로 실행")
    
    # 학습 파라미터 설정 (Stage 3 최적화)
    training_config = {
        'model_size': args.model_size,
        'image_size': args.imgsz,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'device': device,
        'workers': args.workers,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'classes': 45,  # Stage 3: clefG 포함, beam/tie/slur 제외로 45개
        'amp': not args.no_amp,
        'use_wandb': args.wandb,
        'resume': args.resume,
        'project': args.project,
        'name': args.name,
        'mosaic': args.mosaic,
        'gradient_clip': args.gradient_clip
    }
    
    # 설정 요약 출력
    print(f"\n📋 Stage 3 훈련 설정:")
    print(f"   🔧 모델: {training_config['model_size']} (Stage 2: yolov8s → Stage 3: yolov8m)")
    print(f"   📐 이미지 크기: {training_config['image_size']}×{training_config['image_size']}")
    print(f"   📦 배치 크기: {training_config['batch_size']} (안정성 최우선)")
    print(f"   🎯 에폭: {training_config['epochs']}")
    print(f"   📚 학습률: {training_config['lr0']}")
    print(f"   ⚙️  옵티마이저: {training_config['optimizer']}")
    print(f"   🔥 Warmup: {training_config['warmup_epochs']} 에폭")
    print(f"   ✂️  Gradient Clip: {training_config['gradient_clip']}")
    print(f"   🎨 Mosaic: {training_config['mosaic']} (메모리 절약)")
    print(f"   🏷️  클래스 수: {training_config['classes']}개 (clefG 포함)")
    
    # 데이터셋 경로 확인
    dataset_yaml = args.data
    data_path = Path(config['target_path'])
    
    if not Path(dataset_yaml).exists():
        print(f"❌ 데이터셋 설정 파일을 찾을 수 없습니다: {dataset_yaml}")
        print("먼저 python preprocess_stage3_20250729.py를 실행하세요.")
        return
    
    if not data_path.exists():
        print(f"❌ 데이터셋 경로를 찾을 수 없습니다: {data_path}")
        print("먼저 python preprocess_stage3_20250729.py를 실행하세요.")
        return
    
    # 클래스 가중치 계산
    class_weights = calculate_class_weights(data_path)
    
    # W&B 설정
    if training_config['use_wandb']:
        setup_wandb(training_config)
    
    # 모델 로드 (YOLOv8m)
    print(f"📦 모델 로드: {training_config['model_size']} (25.9M 파라미터)")
    model = YOLO(training_config['model_size'])
    
    # Stage 3 개선사항 출력
    print(f"\n🔧 Stage 2 → Stage 3 개선사항:")
    print(f"   ✅ 모델: YOLOv8s (11.2M) → YOLOv8m (25.9M 파라미터)")
    print(f"   ✅ clefG (높은음자리표) 정상 포함")
    print(f"   ✅ beam/tie/slur 정확히 제외 (ID 매핑 수정)")
    print(f"   ✅ 클래스 검증 시스템 추가")
    print(f"   ✅ 배치 {training_config['batch_size']}, Mosaic {training_config['mosaic']} (안정성)")
    print(f"   ✅ 예상 성능: mAP50 0.82-0.85 (Stage 2: 0.75)")
    
    # 학습 실행
    print(f"\n🎯 학습 시작...")
    try:
        results = model.train(
            data=dataset_yaml,
            epochs=training_config['epochs'],
            imgsz=training_config['image_size'],
            batch=training_config['batch_size'],
            device=training_config['device'],
            workers=training_config['workers'],
            optimizer=training_config['optimizer'],
            lr0=training_config['lr0'],
            weight_decay=training_config['weight_decay'],
            warmup_epochs=training_config['warmup_epochs'],
            patience=training_config['patience'],
            save=True,
            save_period=10,  # 10 에폭마다 저장
            val=True,
            plots=True,
            verbose=True,
            # Mixed precision 설정
            amp=training_config['amp'],
            # 데이터 증강 (음악 기호에 적합하게 조정)
            hsv_h=0.01,      # 색조 변화 최소화
            hsv_s=0.5,       # 채도 변화 감소
            hsv_v=0.3,       # 명도 변화 감소
            degrees=0.0,     # 회전 금지 (음악 기호는 방향 중요)
            translate=0.05,  # 이동 최소화
            scale=0.3,       # 스케일 변화 감소
            shear=0.0,       # 전단 변환 금지
            perspective=0.0, # 원근 변환 금지
            flipud=0.0,      # 상하 뒤집기 금지
            fliplr=0.3,      # 좌우 뒤집기 감소
            mosaic=training_config['mosaic'],  # 메모리 절약을 위해 0.0
            mixup=0.0,       # Mixup 비활성화
            copy_paste=0.0,  # Copy-paste 비활성화
            # 프로젝트 설정
            project=training_config['project'],
            name=training_config['name'],
            # 재개 설정
            resume=training_config['resume']
        )
        
        print("🎉 학습 완료!")
        print(f"📊 최고 mAP@0.5: {results.results_dict['metrics/mAP50(B)']:.3f}")
        print(f"📊 최고 mAP@0.5:0.95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")
        
        # 최종 모델 저장 (날짜 포함)
        final_model_name = f'yolov8m_stage3_{datetime.now().strftime("%Y%m%d")}_final.pt'
        model.save(final_model_name)
        print(f"💾 최종 모델 저장: {final_model_name}")
        
        # 학습 통계 저장
        training_stats = {
            'best_mAP50': float(results.results_dict['metrics/mAP50(B)']),
            'best_mAP50_95': float(results.results_dict['metrics/mAP50-95(B)']),
            'total_epochs': training_config['epochs'],
            'final_lr': float(results.results_dict['lr/pg0']),
            'model_size': training_config['model_size'],
            'model_parameters': '25.9M',
            'image_size': training_config['image_size'],
            'batch_size': training_config['batch_size'],
            'classes': training_config['classes'],
            'optimizer': training_config['optimizer'],
            'lr0': training_config['lr0'],
            'improvements_from_stage2': [
                'YOLOv8s → YOLOv8m (성능 향상)',
                'clefG (높은음자리표) 포함',
                'beam/tie/slur 정확히 제외',
                '클래스 검증 시스템 추가'
            ],
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'stage': 'stage3'
        }
        
        stats_file = f'stage3_training_stats_{datetime.now().strftime("%Y%m%d")}.json'
        with open(stats_file, 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        print(f"📈 훈련 통계 저장: {stats_file}")
        return results
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        return None
    
    finally:
        if training_config['use_wandb']:
            wandb.finish()

def validate_stage3(model_path, data_path='data_stage3_20250730/deepscores_stage3.yaml'):
    """Stage 3 모델 검증"""
    print("🔍 Stage 3 모델 검증...")
    
    if not Path(model_path).exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    model = YOLO(model_path)
    
    # 검증 실행
    results = model.val(
        data=data_path,
        imgsz=1024,
        batch=8,  # 검증은 배치 크기를 더 크게
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        save_json=True,
        save_hybrid=True,
        plots=True,
        verbose=True
    )
    
    print("✅ 검증 완료!")
    return results

def main():
    parser = argparse.ArgumentParser(description="Stage 3 YOLOv8m 훈련 - clefG 포함, 고성능 음악 기호 검출")
    parser.add_argument("--validate", action="store_true", help="검증만 실행")
    parser.add_argument("--model", default="yolov8m_stage3_final.pt", help="검증할 모델 경로")
    parser.add_argument("--data", default="data_stage3_20250730/deepscores_stage3.yaml", help="데이터셋 YAML 파일")
    
    # 모델 및 훈련 설정
    parser.add_argument("--model-size", default="yolov8m.pt", choices=["yolov8s.pt", "yolov8m.pt", "yolov8l.pt"], 
                        help="YOLOv8 모델 크기 (기본: yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="학습 에폭 수 (기본: 100)")
    parser.add_argument("--batch-size", type=int, default=1, help="배치 크기 (기본: 1, 안정성 최우선)")
    parser.add_argument("--imgsz", type=int, default=1024, help="입력 이미지 크기 (기본: 1024)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (기본: 20)")
    
    # Stage 3 특화 설정 (성능 최우선)
    parser.add_argument("--lr0", type=float, default=0.001, help="초기 학습률 (기본: 0.001)")
    parser.add_argument("--optimizer", default="SGD", choices=["SGD", "AdamW"], help="옵티마이저 (기본: SGD)")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="가중치 감쇠 (기본: 0.0005)")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup 에폭 수 (기본: 5)")
    parser.add_argument("--gradient-clip", type=float, default=10.0, help="Gradient clipping (기본: 10.0, 0=비활성화)")
    
    # 시스템 설정
    parser.add_argument("--device", default="0", help="CUDA 장치 (기본: 0, CPU는 'cpu')")
    parser.add_argument("--workers", type=int, default=4, help="데이터로더 워커 수 (기본: 4)")
    
    # 기능 설정
    parser.add_argument("--no-amp", action="store_true", help="Mixed precision 비활성화")
    parser.add_argument("--wandb", action="store_true", default=True, help="W&B 로깅 활성화 (기본: 활성화)")
    parser.add_argument("--resume", action="store_true", help="마지막 체크포인트에서 재개")
    parser.add_argument("--mosaic", type=float, default=0.0, help="Mosaic 증강 확률 (기본: 0.0, 메모리 절약)")
    
    # 프로젝트 설정
    parser.add_argument("--project", default="runs/stage3", help="프로젝트 디렉토리 (기본: runs/stage3)")
    parser.add_argument("--name", default=f"yolov8m_45classes_{datetime.now().strftime('%m%d_%H%M')}", 
                        help="실행 이름")
    
    args = parser.parse_args()
    
    if args.validate:
        validate_stage3(args.model, args.data)
    else:
        train_stage3(args)

if __name__ == "__main__":
    main()