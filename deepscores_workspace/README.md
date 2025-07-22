# DeepScores YOLOv8 파이프라인

이 디렉터리는 DeepScores 데이터셋을 사용하여 YOLOv8 음표 인식 모델을 학습하기 위한 완전한 파이프라인을 제공합니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필요한 라이브러리 설치
pip install ultralytics wandb scikit-learn pyyaml tqdm optuna streamlit

# 또는 프로젝트 루트에서
pip install -r requirements.txt
```

### 2. 데이터 준비

DeepScores V2 데이터셋을 다운로드하고 `raw_data/` 디렉터리에 압축 해제:

```
raw_data/
├── images_png/          # 이미지 파일들
├── annotations/         # NPZ 어노테이션 파일들
└── ...
```

### 3. Pilot Mode 실행 (권장)

전체 데이터셋 처리 전에 작은 샘플로 파이프라인을 검증:

```bash
# 1,000개 샘플로 전처리
python preprocess_deepscores.py --pilot-mode --sample-size 1000

# 조기 검증 실행
python early_validation.py

# Pilot 학습 (빠른 테스트)
python train_yolo.py --data data/deepscores.yaml --epochs 5 --model yolov8n --name pilot_test
```

### 4. 본격 학습

Pilot Mode가 성공하면 전체 파이프라인 실행:

```bash
# 전체 데이터 전처리
python preprocess_deepscores.py

# 점진적 학습 실행
python train_yolo.py --data data/deepscores.yaml --progressive --model yolov8s

# 또는 일반 학습
python train_yolo.py --data data/deepscores.yaml --epochs 100 --model yolov8s
```

## 📁 디렉터리 구조

```
deepscores_workspace/
├── raw_data/                    # DeepScores 원본 데이터
│   ├── images_png/
│   └── annotations/
├── data/                        # 변환된 YOLO 형식 데이터
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── deepscores.yaml
├── validation_reports/          # 데이터 품질 검증 리포트
├── runs/                        # 학습 결과 (자동 생성)
│   └── detect/
│       └── train*/
│           └── weights/
│               ├── best.pt      # 최종 모델
│               └── last.pt
├── early_validation.py          # 조기 검증 시스템
├── preprocess_deepscores.py     # 데이터 전처리
├── train_yolo.py               # 학습 스크립트
├── symbol_detector.py          # 추론 스크립트
└── README.md                   # 이 파일
```

## 🛠️ 주요 스크립트

### `preprocess_deepscores.py`

DeepScores NPZ 형식을 YOLO TXT 형식으로 변환하고 품질 검증 수행:

```bash
# 기본 사용법
python preprocess_deepscores.py --raw-data raw_data --output data

# Pilot Mode (샘플 데이터만)
python preprocess_deepscores.py --pilot-mode --sample-size 1000

# 옵션:
#   --raw-data: 원본 데이터 디렉터리
#   --output: 출력 디렉터리  
#   --pilot-mode: 샘플 데이터만 처리
#   --sample-size: 샘플 크기 (기본: 1000)
```

**출력:**
- `data/`: YOLO 형식 데이터셋
- `validation_reports/`: 품질 검증 리포트 (JSON, PNG)

### `early_validation.py`

파이프라인 건전성 조기 검증:

```bash
python early_validation.py --workspace .

# 검증 항목:
# 1. 데이터 로딩 테스트
# 2. 모델 초기화 테스트  
# 3. 1 epoch 학습 테스트
# 4. 추론 테스트
```

### `train_yolo.py`

고급 기능을 포함한 YOLOv8 학습:

```bash
# 점진적 학습 (권장)
python train_yolo.py --data data/deepscores.yaml --progressive --model yolov8s

# 일반 학습
python train_yolo.py --data data/deepscores.yaml --epochs 100 --batch 8

# 하이퍼파라미터 최적화
python train_yolo.py --data data/deepscores.yaml --optimize

# 주요 옵션:
#   --progressive: 점진적 클래스 확장 학습
#   --optimize: Optuna 하이퍼파라미터 최적화
#   --model: yolov8n/s/m/l/x 
#   --batch: 배치 크기 또는 'auto'
#   --no-wandb: W&B 비활성화
```

### `symbol_detector.py`

학습된 모델로 추론 수행:

```bash
# 단일 이미지
python symbol_detector.py --model runs/detect/train/weights/best.pt --image test.png --visualize

# 배치 처리
python symbol_detector.py --model best.pt --batch images_folder/ --output results/ --visualize

# Tiled inference (큰 이미지)
python symbol_detector.py --model best.pt --image large_image.png --tiled --tile-size 1024

# 옵션:
#   --conf: 신뢰도 임계값 (기본: 0.5)
#   --device: auto/cpu/cuda/mps
#   --visualize: 결과 시각화 저장
```

## 📊 학습 모니터링

### Weights & Biases 연동

```bash
# W&B 로그인 (최초 1회)
wandb login

# 학습 시 자동으로 W&B에 기록됨
python train_yolo.py --data data/deepscores.yaml --epochs 50
```

모니터링 메트릭:
- Training/Validation Loss
- mAP@0.5, mAP@0.75
- 클래스별 Precision/Recall
- GPU 메모리 사용량
- 샘플 예측 결과

### 로컬 결과 확인

```bash
# TensorBoard (선택사항)
tensorboard --logdir runs/detect

# 결과 파일 위치
ls runs/detect/train*/
# ├── weights/best.pt
# ├── results.png           # 학습 곡선
# ├── confusion_matrix.png  # 혼동 행렬
# └── val_batch*.jpg       # 검증 샘플 결과
```

## 🎯 점진적 학습 전략

현재 구현된 3단계 점진적 학습:

1. **Phase 1**: 핵심 클래스 (noteheadFull, stem, gClef)
2. **Phase 2**: 리듬 요소 (restQuarter, beam, dot)  
3. **Phase 3**: 임시표 (sharp, flat, natural)

각 단계별 목표:
- Phase 1: mAP@0.5 ≥ 0.85
- Phase 2: mAP@0.5 ≥ 0.80  
- Phase 3: mAP@0.5 ≥ 0.75

## 🔧 문제 해결

### 메모리 부족 오류

```bash
# 배치 크기 줄이기
python train_yolo.py --data data/deepscores.yaml --batch 4

# 이미지 크기 줄이기  
python train_yolo.py --data data/deepscores.yaml --imgsz 640

# 작은 모델 사용
python train_yolo.py --data data/deepscores.yaml --model yolov8n
```

### 데이터 문제

```bash
# 데이터 품질 재검증
python preprocess_deepscores.py --pilot-mode --sample-size 100

# 검증 리포트 확인
ls validation_reports/
```

### 학습이 진행되지 않음

```bash
# 조기 검증으로 문제점 파악
python early_validation.py

# 매우 간단한 테스트 (1 epoch)
python train_yolo.py --data data/deepscores.yaml --epochs 1 --model yolov8n
```

## 📈 성능 벤치마크

예상 성능 (YOLOv8s, RTX 3080 기준):

| 단계 | 클래스 수 | mAP@0.5 | 학습 시간 | GPU 메모리 |
|------|-----------|---------|-----------|------------|
| Phase 1 | 3 | 0.85+ | 2-3시간 | 6GB |
| Phase 2 | 6 | 0.80+ | 3-4시간 | 6GB |  
| Phase 3 | 9 | 0.75+ | 4-5시간 | 7GB |

추론 속도:
- 1024x1024 이미지: ~50ms (RTX 3080)
- CPU: ~500ms

## 🔄 백업 전략

주 전략 실패 시 자동으로 대안 선택:

1. **Primary**: YOLOv8s + Progressive Training
2. **Backup 1**: YOLOv8n + All-class Training  
3. **Backup 2**: YOLOv8s + Reduced Epochs
4. **Emergency**: Pre-trained YOLO + Rule-based

## 🚀 다음 단계

학습 완료 후:

1. **모델 통합**: `../symbol_detector.py`를 ScoreEye에 통합
2. **성능 테스트**: 실제 악보 이미지로 검증
3. **최적화**: ONNX 변환, TensorRT 적용
4. **확장**: 추가 클래스 학습

---

**문의사항이나 오류 발생 시:**
1. `validation_reports/`의 검증 리포트 확인
2. W&B 대시보드에서 학습 로그 분석  
3. GitHub Issues에 상세한 오류 로그와 함께 문의