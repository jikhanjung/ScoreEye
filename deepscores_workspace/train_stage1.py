#!/usr/bin/env python3
"""
Stage 1 YOLOv8 학습 스크립트
50개 핵심 클래스로 제한한 모델 학습
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO
import torch
import wandb
import yaml

def setup_wandb(config):
    """Weights & Biases 설정"""
    wandb.init(
        project="deepscores-stage1",
        name=f"yolov8s-stage1-{config['image_size']}px",
        config={
            'model': 'yolov8s',
            'classes': config['classes'],
            'image_size': config['image_size'],
            'batch_size': config['batch_size'],
            'epochs': config['epochs'],
            'device': config['device']
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
    
    # 클래스별 빈도
    class_counts = {}
    for cls in class_info['classes']:
        class_counts[cls['id']] = cls['count']
    
    # 가중치 계산 (역빈도)
    total_samples = sum(class_counts.values())
    class_weights = {}
    
    for i, cls in enumerate(class_info['classes']):
        original_id = cls['id']
        frequency = cls['count'] / total_samples
        # 가중치 = 1 / sqrt(frequency) (너무 극단적이지 않게)
        weight = 1.0 / (frequency ** 0.5)
        class_weights[i] = weight
    
    print(f"✅ 클래스 가중치 계산 완료")
    return class_weights

def create_custom_loss():
    """커스텀 손실 함수 생성 (클래스 불균형 해결)"""
    # YOLOv8은 내부적으로 focal loss를 사용하므로
    # 추가 가중치는 학습 시 class_weight 파라미터로 전달
    pass

def train_stage1(args):
    """Stage 1 모델 학습"""
    print("🚀 Stage 1 YOLOv8 학습 시작")
    
    # 설정 로드
    with open('stage1_preprocess_config.json', 'r') as f:
        config = json.load(f)
    
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
    
    # 학습 파라미터 설정 (args로부터)
    training_config = {
        'model_size': args.model_size,
        'image_size': args.imgsz,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'device': device,
        'workers': args.workers,
        'optimizer': 'AdamW',
        'lr0': args.lr0,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'classes': 50,
        'amp': not args.no_amp,
        'use_wandb': args.wandb,
        'resume': args.resume,
        'project': args.project,
        'name': args.name
    }
    
    # 데이터셋 경로 확인
    dataset_yaml = 'deepscores_stage1.yaml'
    if not Path(dataset_yaml).exists():
        print(f"❌ 데이터셋 설정 파일을 찾을 수 없습니다: {dataset_yaml}")
        return
    
    data_path = Path(config['target_path'])
    if not data_path.exists():
        print(f"❌ 데이터셋 경로를 찾을 수 없습니다: {data_path}")
        print("먼저 python preprocess_stage1.py를 실행하세요.")
        return
    
    # 클래스 가중치 계산
    class_weights = calculate_class_weights(data_path)
    
    # W&B 설정
    if training_config['use_wandb']:
        setup_wandb(training_config)
    
    # 모델 로드
    print(f"📦 모델 로드: {training_config['model_size']}")
    model = YOLO(training_config['model_size'])
    
    # 학습 실행
    print("🎯 학습 시작...")
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
            # 데이터 증강
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,  # 음악 기호는 회전하면 안 됨
            translate=0.1,
            scale=0.5,
            shear=0.0,  # 음악 기호는 전단하면 안 됨
            perspective=0.0,  # 음악 기호는 원근 변환하면 안 됨
            flipud=0.0,  # 상하 뒤집기 금지
            fliplr=0.5,  # 좌우 뒤집기는 일부 허용
            mosaic=args.mosaic,
            mixup=0.0,  # 음악 기호에는 mixup 부적합
            copy_paste=0.0,  # copy-paste 증강 비활성화
            # 프로젝트 설정
            project=training_config['project'],
            name=training_config['name'],
            # 재개 설정
            resume=training_config['resume']
        )
        
        print("🎉 학습 완료!")
        print(f"📊 최고 mAP@0.5: {results.results_dict['metrics/mAP50(B)']:.3f}")
        print(f"📊 최고 mAP@0.5:0.95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")
        
        # 최종 모델 저장
        model.save('yolov8s_stage1_final.pt')
        print("💾 최종 모델 저장: yolov8s_stage1_final.pt")
        
        # 학습 통계 저장
        training_stats = {
            'best_mAP50': float(results.results_dict['metrics/mAP50(B)']),
            'best_mAP50_95': float(results.results_dict['metrics/mAP50-95(B)']),
            'total_epochs': training_config['epochs'],
            'final_lr': float(results.results_dict['lr/pg0']),
            'model_size': training_config['model_size'],
            'image_size': training_config['image_size'],
            'batch_size': training_config['batch_size'],
            'classes': training_config['classes']
        }
        
        with open('stage1_training_stats.json', 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        return results
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        return None
    
    finally:
        if training_config['use_wandb']:
            wandb.finish()

def validate_stage1(model_path='yolov8s_stage1_final.pt'):
    """Stage 1 모델 검증"""
    print("🔍 Stage 1 모델 검증...")
    
    if not Path(model_path).exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    model = YOLO(model_path)
    
    # 검증 실행
    results = model.val(
        data='deepscores_stage1.yaml',
        imgsz=1536,
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
    parser = argparse.ArgumentParser(description="Stage 1 YOLOv8 학습")
    parser.add_argument("--validate", action="store_true", help="검증만 실행")
    parser.add_argument("--model", default="yolov8s_stage1_final.pt", help="검증할 모델 경로")
    parser.add_argument("--model-size", default="yolov8s.pt", choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], 
                        help="YOLOv8 모델 크기 (기본: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=120, help="학습 에폭 수 (기본: 120)")
    parser.add_argument("--batch-size", type=int, default=3, help="배치 크기 (기본: 3)")
    parser.add_argument("--imgsz", type=int, default=1536, help="입력 이미지 크기 (기본: 1536)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (기본: 20)")
    parser.add_argument("--lr0", type=float, default=0.01, help="초기 학습률 (기본: 0.01)")
    parser.add_argument("--device", default="0", help="CUDA 장치 (기본: 0, CPU는 'cpu')")
    parser.add_argument("--workers", type=int, default=4, help="데이터로더 워커 수 (기본: 4)")
    parser.add_argument("--no-amp", action="store_true", help="Mixed precision 비활성화")
    parser.add_argument("--wandb", action="store_true", default=True, help="W&B 로깅 활성화 (기본: 활성화)")
    parser.add_argument("--resume", action="store_true", help="마지막 체크포인트에서 재개")
    parser.add_argument("--project", default="runs/stage1", help="프로젝트 디렉토리 (기본: runs/stage1)")
    parser.add_argument("--name", default="yolov8s_50classes", help="실행 이름")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic 증강 확률 (기본: 1.0, 0=비활성화)")
    
    args = parser.parse_args()
    
    if args.validate:
        validate_stage1(args.model)
    else:
        train_stage1(args)

if __name__ == "__main__":
    main()