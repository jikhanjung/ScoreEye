### **종합 YOLOv8 음표 인식 모델 구현 계획 (DeepScores 기반)**

**문서 목적**: DeepScores 데이터셋을 활용하여 프로덕션 레벨의 YOLOv8 음표 인식 모델을 구축하기 위한 종합적이고 실행 가능한 계획을 수립한다. 이 문서는 초기 계획과 전문가의 검토 및 제안 사항을 통합한 최종 버전이다.

---

### **🎯 최종 목표**

1.  **고성능 모델 확보**: DeepScores 데이터셋으로 학습하고 실제 악보에 파인튜닝된, 정확하고 강건한 YOLOv8 모델(`best.pt`)을 확보한다.
2.  **프로젝트 통합**: 완성된 모델을 `ScoreEye` 프로젝트에 통합하여, 추출된 마디 이미지 내의 개별 악보 기호를 감지하고, 그 결과를 후속 처리(예: MusicXML 변환)에 사용할 수 있도록 구현한다.

---

### **Phase 1: 환경 설정 및 데이터 준비 (2일)**

1.  **필수 라이브러리 설치**:
    - `requirements.txt`에 `ultralytics`, `numpy`, `opencv-python`, `pyyaml`, `tqdm`, `scikit-learn`, `wandb` 등을 추가하고 설치한다.

2.  **DeepScores 데이터셋 다운로드**:
    - 공식 웹사이트에서 "DeepScoresV2 (Main dataset)"를 다운로드하여 `deepscores_workspace/raw_data/`에 압축 해제한다.

3.  **프로젝트 디렉터리 구조 설정**:
    ```
    ScoreEye/
    ├── deepscores_workspace/
    │   ├── data/                 # YOLOv8 학습용 데이터셋
    │   ├── raw_data/             # DeepScores 원본 데이터
    │   ├── validation_reports/   # 데이터 품질 검증 리포트 저장
    │   ├── preprocess_deepscores.py # 데이터 변환 및 검증 스크립트
    │   └── symbol_detector.py    # 추론 스크립트
    ├── models/                   # 학습 완료된 모델 가중치 저장
    └── ...
    ```

---

### **Phase 2: 데이터 전처리 및 품질 검증 (5일)**

**핵심 목표**: 단순 포맷 변환을 넘어, 데이터의 품질을 보장하고 잠재적 문제를 사전에 해결한다.

1.  **`preprocess_deepscores.py` 스크립트 기능 확장**:
    - **(기본)** NPZ 어노테이션을 YOLO TXT 포맷으로 변환.
    - **(추가)** **데이터 무결성 검증**: 이미지-라벨 쌍이 누락되지 않았는지, Bounding Box 좌표가 유효한지(0~1 범위, 너비/높이 > 0) 검사.
    - **(추가)** **클래스 분포 분석**: 135개 전체 클래스 및 목표 클래스의 분포를 분석하고, 심각한 불균형(예: `noteheadFull` 90%, `dot` 1%)을 식별하여 리포트 생성. 이 리포트는 학습 시 가중치 손실(weighted loss) 적용의 근거가 된다.
    - **(추가)** **시각적 샘플 검증**: 무작위로 100개의 이미지를 샘플링하여 Bounding Box와 라벨을 시각화한 HTML 리포트를 생성. 이를 통해 미묘한 라벨링 오류를 수동으로 검토.

2.  **실행 프로세스**:
    1.  `preprocess_deepscores.py`를 실행하여 데이터 변환 및 자동 검증을 수행.
    2.  생성된 `validation_reports/`의 클래스 분포 리포트와 시각적 샘플 리포트를 검토.
    3.  데이터에 심각한 문제가 발견되면, 해당 데이터를 제외하거나 수정하는 규칙을 스크립트에 추가하고 재실행.

---

### **Phase 3: 모델 학습 및 최적화 (12일)**

**핵심 목표**: 점진적 학습, 도메인 적응, 메모리 최적화 전략을 사용하여 안정적이고 효율적으로 모델을 학습시킨다.

1.  **점진적 클래스 확장 전략 (Progressive Class Expansion)**:
    - 한 번에 모든 클래스를 학습하는 대신, 단계적으로 확장하며 모델을 안정적으로 고도화한다.
    - **Phase 3.1 (3-4일)**: 핵심 3개 클래스(`noteheadFull`, `stem`, `gClef`)로 기본 모델 학습.
    - **Phase 3.2 (2-3일)**: 이전 모델 가중치를 이어받아 기본 리듬 요소(`restQuarter`, `beam`, `dot`) 추가 학습.
    - **Phase 3.3 (2일)**: 임시표(`sharp`, `flat`, `natural`) 추가 학습.
    - **Phase 3.4 (2일)**: 조표/박자표(`timeSig4_4`, `keySigFlat1` 등) 추가 학습.

2.  **도메인 적응 전략 (Domain Adaptation)**:
    - 합성 데이터(DeepScores)와 실제 데이터 간의 간극을 해소한다.
    - **Pre-training**: DeepScores 원본 데이터로 각 클래스 확장 단계의 기본 학습을 진행.
    - **Noise Simulation**: 학습된 모델에 노이즈, 기울어짐, 명암 변화 등 Augmentation이 적용된 데이터로 추가 학습하여 강건성 확보.
    - **Fine-tuning**: `ScoreEye`로 실제 스캔한 악보 중 일부를 수동 라벨링하여 최종적으로 미세 조정. (약 100~200개 이미지)

3.  **메모리 및 성능 최적화**:
    - **`deepscores.yaml` 작성**: 학습 데이터 경로와 클래스 이름 정의.
    - **학습 명령어**: 다음 전략들을 조합하여 저사양 GPU에서도 학습이 가능하도록 설정.
      ```bash
      # 예시: Gradient Accumulation과 Mixed Precision 사용
      yolo task=detect mode=train model=yolov8s.pt data=... \
           epochs=100 imgsz=1024 batch=4 patience=15 \
           amp=True # Mixed Precision (FP16)
      # Note: Gradient Accumulation은 코드 레벨에서 설정 필요할 수 있음
      ```

4.  **성능 모니터링 및 추적 (W&B 연동)**:
    - 모든 학습 과정은 `wandb`와 연동하여 실험 결과를 체계적으로 기록.
    - `training_loss`, `validation_mAP`, 클래스별 정밀도/재현율, GPU 사용량 등을 실시간으로 추적하여 최적의 모델을 선정.

---

### **Phase 4: 성능 평가 및 실패 분석 (5일)**

**핵심 목표**: 표준 메트릭과 OMR 특화 메트릭을 모두 사용하여 모델을 다각도로 평가하고, 실패 사례 분석을 통해 개선 방향을 도출한다.

1.  **종합 성능 평가 시스템 구축**:
    - **표준 메트릭**: `mAP@0.5`, `mAP@0.75` 및 클래스별 `Precision`, `Recall`, `F1-score`를 계산.
    - **OMR 특화 메트릭**: 
        - **음표 완성도**: `notehead`와 `stem`이 올바르게 조합되는 비율.
        - **수직 정렬 정확도**: 동일 시간대의 음표들이 수직으로 정렬되는지 평가.
        - **클래스 혼동 행렬**: `noteheadFull`과 `noteheadHalf` 등 유사 클래스 간의 혼동을 분석.

2.  **실패 사례 분석 및 시각화 도구 개발**:
    - **자동 실패 분류**: 예측 결과를 Ground Truth와 비교하여 `False Positive`, `False Negative`, `Misclassification` 등 유형별로 실패 사례를 자동 분류.
    - **시각화 대시보드 (Streamlit/Flask)**: 분류된 실패 사례를 인터랙티브하게 탐색할 수 있는 웹 대시보드 구축. (예: 특정 클래스의 실패 사례만 필터링, 신뢰도 점수별 분석 등)
    - **자동 개선 제안**: 분석 결과를 바탕으로 "`restHalf` 클래스의 재현율이 낮으니, 해당 클래스에 대한 데이터 증강이 필요합니다." 와 같은 개선 방향을 자동으로 제안하는 시스템 구현.

---

### **Phase 5: 추론 및 프로젝트 통합 (4일)**

1.  **최적화된 추론 스크립트 작성 (`symbol_detector.py`)**:
    - 최종 선택된 모델(`best.pt`)을 로드.
    - **Tiled Inference 구현**: 고해상도 마디 이미지를 작은 타일로 나누어 추론하고 결과를 병합하여, 추론 시 메모리 사용량을 최소화.
    - 감지된 기호의 `box`, `class_name`, `confidence`를 구조화된 형식으로 반환.

2.  **`ScoreEye` 파이프라인 통합**:
    - `extract_measures.py` 또는 `scoreeye_gui.py`에서 마디 이미지를 생성한 후, `SymbolDetector`를 호출.
    - 반환된 기호의 상대 좌표를 마디의 `metadata.json`과 결합하여 페이지 전체에서의 절대 좌표로 변환.
    - 이 최종 정보를 MusicXML 생성 모듈의 입력으로 전달.

---

### **🗓️ 예상 타임라인 (총 6주)**

- **Week 1-2: 데이터 준비 및 검증**
  - [ ] 환경 설정, 데이터 다운로드 (2일)
  - [ ] 데이터 품질 검증 시스템 구축 및 실행 (3일)
  - [ ] 클래스 불균형 분석 및 해결 전략 수립 (2일)

- **Week 3-4: 모델 학습 및 평가 시스템 구축**
  - [ ] 성능 평가 메트릭 시스템 구현 (2일)
  - [ ] 점진적 학습 Phase 1, 2 진행 (5일)
  - [ ] 실패 사례 분석 도구 프로토타입 개발 (3일)

- **Week 5-6: 모델 고도화 및 통합**
  - [ ] 점진적 학습 Phase 3, 4 및 도메인 적응 파인튜닝 진행 (4일)
  - [ ] `symbol_detector.py` 최적화 및 프로젝트 통합 (3일)
  - [ ] 종합 테스트 및 성능 모니터링 대시보드 완성 (3일)

---

### **🔧 추가 개선 제안 및 위험 완화 방안**

#### **1. 조기 검증 체계 (Early Validation System)**

각 Phase 완료 후 전체 파이프라인의 건전성을 조기에 검증하여 후속 작업의 실패 위험을 최소화합니다.

```python
def early_pipeline_validation():
    """Phase 2 완료 후 즉시 실행하는 파이프라인 검증"""
    
    validation_steps = [
        {
            'name': 'data_loading_test',
            'description': '10개 샘플 이미지와 라벨이 정상적으로 로드되는지 확인',
            'expected_time': '< 30초'
        },
        {
            'name': 'model_initialization_test', 
            'description': 'YOLOv8s 모델이 GPU에서 정상 초기화되는지 확인',
            'expected_time': '< 1분'
        },
        {
            'name': 'single_epoch_training_test',
            'description': '1 epoch 학습이 오류 없이 완료되는지 확인',
            'expected_time': '< 10분'
        },
        {
            'name': 'inference_test',
            'description': '학습된 모델로 샘플 이미지 추론이 가능한지 확인',
            'expected_time': '< 1분'
        }
    ]
    
    # 모든 단계 통과 시에만 Phase 3 진행 허용
    for step in validation_steps:
        try:
            execute_validation_step(step)
            print(f"✅ {step['name']} 통과")
        except Exception as e:
            print(f"❌ {step['name']} 실패: {e}")
            return False  # 즉시 중단하고 문제 해결
    
    return True  # 모든 검증 통과
```

#### **2. 다층 백업 전략 (Multi-Level Backup Strategy)**

주 전략 실패 시를 대비한 체계적인 백업 계획을 수립합니다.

```python
backup_strategies = {
    'primary': {
        'model': 'YOLOv8s',
        'strategy': 'Progressive Class Expansion',
        'expected_performance': 'mAP@0.5: 0.80',
        'training_time': '12일',
        'gpu_requirement': '8GB+'
    },
    'backup_level_1': {
        'model': 'YOLOv8n',  # 더 가벼운 모델
        'strategy': 'All-class simultaneous training',
        'expected_performance': 'mAP@0.5: 0.70',
        'training_time': '6일',
        'gpu_requirement': '4GB+'
    },
    'backup_level_2': {
        'model': 'Detectron2 ResNet-50',
        'strategy': 'Traditional CNN approach',
        'expected_performance': 'mAP@0.5: 0.75',
        'training_time': '8일',
        'gpu_requirement': '6GB+'
    },
    'emergency_fallback': {
        'model': 'Pre-trained YOLO + Rule-based post-processing',
        'strategy': 'Minimal custom training',
        'expected_performance': 'mAP@0.5: 0.50',
        'training_time': '2일',
        'gpu_requirement': '2GB+'
    }
}

def select_backup_strategy(failure_reason):
    """실패 원인에 따른 최적 백업 전략 선택"""
    if failure_reason == 'gpu_memory_insufficient':
        return backup_strategies['backup_level_1']
    elif failure_reason == 'training_time_constraint':
        return backup_strategies['emergency_fallback'] 
    elif failure_reason == 'convergence_failure':
        return backup_strategies['backup_level_2']
    else:
        return backup_strategies['backup_level_1']
```

#### **3. 정량적 성공 기준 및 품질 관리 (Quantitative Success Criteria)**

프로젝트 성공 여부를 객관적으로 판단할 수 있는 구체적 기준을 설정합니다.

```yaml
# success_criteria.yaml
performance_benchmarks:
  minimum_acceptable:
    overall_mAP@0.5: 0.60
    per_class_minimum_f1: 0.50  # 모든 클래스가 최소 F1 0.5 이상
    inference_speed: '<100ms per 1024x1024 measure'
    memory_usage: '<4GB GPU memory'
    integration_success_rate: 0.95  # ScoreEye 파이프라인 통합 성공률
    
  target_performance:
    overall_mAP@0.5: 0.80
    key_classes_f1: 0.85  # noteheadFull, stem, gClef 등 핵심 클래스
    inference_speed: '<50ms per measure'
    memory_usage: '<2GB GPU memory'
    integration_success_rate: 0.99
    
  quality_gates:
    phase_3_gate: 'mAP@0.5 >= 0.70 for Phase 3.1-3.2'
    phase_4_gate: 'F1 score >= 0.60 for all target classes'
    final_gate: 'End-to-end pipeline success rate >= 0.90'

# 자동화된 품질 관리
def evaluate_quality_gate(phase, current_metrics):
    """각 Phase 완료 시 품질 게이트 자동 평가"""
    criteria = success_criteria['quality_gates'][f'{phase}_gate']
    
    if meets_criteria(current_metrics, criteria):
        log_success(phase, current_metrics)
        return True
    else:
        log_failure(phase, current_metrics)
        suggest_improvement_actions(current_metrics)
        return False  # 다음 Phase 진행 불허
```

#### **4. 점진적 데이터셋 확장 전략 (Progressive Dataset Scaling)**

DeepScores V2의 거대한 용량(300GB+) 문제를 해결하기 위한 단계적 데이터 활용 방안입니다.

```python
dataset_scaling_plan = {
    'pilot_phase': {
        'size': '1GB (약 1,000개 이미지)',
        'purpose': '파이프라인 검증 및 디버깅',
        'duration': '1일',
        'expected_mAP': '0.40-0.60'
    },
    'development_phase': {
        'size': '10GB (약 10,000개 이미지)', 
        'purpose': '핵심 클래스 학습 및 최적화',
        'duration': '3-4일',
        'expected_mAP': '0.60-0.75'
    },
    'production_phase': {
        'size': '50-100GB (약 50,000-100,000개 이미지)',
        'purpose': '최종 고성능 모델 학습',
        'duration': '7-10일',
        'expected_mAP': '0.75-0.85'
    }
}

def smart_dataset_sampling():
    """클래스 균형을 고려한 지능적 데이터 샘플링"""
    
    # 1. 전체 데이터셋의 클래스 분포 분석
    class_distribution = analyze_full_dataset()
    
    # 2. 목표 클래스별 최소 샘플 수 보장
    min_samples_per_class = 500
    
    # 3. 균형잡힌 샘플링 전략
    sampling_strategy = {
        'rare_classes': 'oversample',      # dot, rest 등 희귀 클래스
        'common_classes': 'undersample',   # noteheadFull 등 과다 클래스
        'balanced_classes': 'normal_sample' # 적절한 분포의 클래스
    }
    
    return generate_balanced_subset(class_distribution, sampling_strategy)
```

#### **5. 실시간 모니터링 및 알림 시스템 (Real-time Monitoring & Alert System)**

학습 과정의 이상 징후를 조기에 감지하고 대응할 수 있는 모니터링 체계입니다.

```python
def setup_advanced_monitoring():
    """W&B 기반 고급 모니터링 시스템 구축"""
    
    # W&B 설정
    wandb.init(
        project="scoreeye-yolov8-production",
        config={
            "dataset_version": "DeepScores-v2-subset",
            "model_architecture": "YOLOv8s", 
            "training_strategy": "progressive_class_expansion",
            "target_classes": 11
        }
    )
    
    # 실시간 알림 조건 설정
    alert_conditions = {
        'training_stalled': {
            'condition': 'validation_loss unchanged for 10 epochs',
            'action': 'adjust learning rate or early stopping'
        },
        'overfitting_detected': {
            'condition': 'train_loss < 0.1 but val_loss > 0.5',
            'action': 'increase regularization or reduce model complexity'
        },
        'gpu_memory_warning': {
            'condition': 'gpu_memory_usage > 90%',
            'action': 'reduce batch size or switch to gradient accumulation'
        },
        'class_imbalance_severe': {
            'condition': 'any class F1 score < 0.3 after 20 epochs',
            'action': 'apply class-specific data augmentation'
        }
    }
    
    # Slack/Email 알림 연동
    def send_alert(condition, metrics):
        message = f"⚠️ Alert: {condition} detected\nCurrent metrics: {metrics}\nRecommended action: {alert_conditions[condition]['action']}"
        # send_to_slack(message) or send_email(message)

# 자동화된 하이퍼파라미터 튜닝
def automated_hyperparameter_optimization():
    """Optuna 기반 자동 하이퍼파라미터 최적화"""
    
    import optuna
    
    def objective(trial):
        # 하이퍼파라미터 공간 정의
        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
        
        # 모델 학습 및 평가
        model_performance = train_and_evaluate_model(lr, batch_size, weight_decay)
        return model_performance['mAP@0.5']
    
    # 최적화 실행 (각 Phase별로)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, timeout=7200)  # 2시간 제한
    
    return study.best_params
```

#### **6. 비용 최적화 및 클라우드 전략 (Cost Optimization & Cloud Strategy)**

제한된 리소스에서 최대 효율을 얻기 위한 실용적 방안입니다.

```python
cloud_optimization_strategy = {
    'local_development': {
        'phase': 'Phase 1-2 (데이터 전처리 및 검증)',
        'resources': 'Local CPU + 작은 GPU',
        'estimated_cost': '$0',
        'duration': '3-4일'
    },
    'cloud_training': {
        'phase': 'Phase 3 (집중 모델 학습)',
        'resources': 'AWS p3.2xlarge (V100 16GB) spot instance',
        'estimated_cost': '$50-100',
        'duration': '8-10일',
        'cost_optimization': [
            'Spot instance 사용으로 70% 비용 절약',
            'S3에 checkpoint 자동 저장',
            'Preemption 대비 재시작 자동화'
        ]
    },
    'inference_optimization': {
        'phase': 'Phase 5 (추론 및 통합)',
        'resources': 'Local GPU 또는 CPU-only',
        'model_optimization': [
            'ONNX 변환으로 추론 속도 2x 향상',
            'TensorRT 최적화 (NVIDIA GPU)',
            'Quantization으로 모델 크기 50% 감소'
        ]
    }
}

def implement_cost_optimization():
    """비용 최적화 구현"""
    
    # 1. 자동 인스턴스 관리
    def auto_spot_instance_management():
        """Spot instance 중단 대비 자동 복구"""
        while training_in_progress:
            try:
                continue_training()
            except SpotInstanceTermination:
                save_checkpoint()
                restart_on_new_instance()
                
    # 2. 모델 경량화
    def optimize_model_for_deployment():
        """추론 최적화"""
        model_optimizations = [
            ('onnx_conversion', lambda: convert_to_onnx()),
            ('pruning', lambda: prune_model_weights()),
            ('quantization', lambda: quantize_model())
        ]
        
        for opt_name, opt_func in model_optimizations:
            try:
                optimized_model = opt_func()
                validate_optimized_model(optimized_model)
                print(f"✅ {opt_name} 최적화 완료")
            except Exception as e:
                print(f"⚠️ {opt_name} 최적화 실패: {e}")
```

#### **7. 수정된 종합 타임라인 (6주 → 7주)**

위험 완화 방안들을 반영한 보다 안전한 타임라인입니다.

- **Week 0 (Preparation)**: ⭐ 신규 추가
  - [ ] 조기 검증 시스템 구축 (1일)
  - [ ] 클라우드 환경 설정 및 비용 최적화 구현 (1일)
  - [ ] Pilot Phase 데이터셋 (1GB) 준비 및 파이프라인 검증 (1일)

- **Week 1-2: 안정화된 데이터 준비**
  - [ ] 환경 설정, 전체 데이터 다운로드 (2일)
  - [ ] 품질 검증 + Development Phase 데이터셋 (10GB) 구축 (3일)
  - [ ] 클래스 불균형 분석 및 스마트 샘플링 전략 구현 (2일)

- **Week 3-4: 핵심 모델 개발**
  - [ ] 모니터링 시스템 + 성능 메트릭 구현 (2일)
  - [ ] Phase 3.1-3.2 학습 + 자동 하이퍼파라미터 최적화 (5일)
  - [ ] 실패 분석 도구 + 품질 게이트 검증 (3일)

- **Week 5-6: 고도화 및 최적화**
  - [ ] Phase 3.3-3.4 + 도메인 적응 파인튜닝 (4일)
  - [ ] Production Phase 데이터셋 (50GB) 최종 학습 (3일)  ⭐ 신규
  - [ ] 모델 경량화 및 추론 최적화 (3일)

- **Week 7: 통합 및 배포**  ⭐ 신규 추가
  - [ ] ScoreEye 파이프라인 통합 및 End-to-End 테스트 (3일)
  - [ ] 성능 벤치마킹 및 최종 검증 (2일)
  - [ ] 문서화 및 배포 준비 (2일)

**예상 총 소요 기간**: **7주** (리스크 버퍼 1주 추가)
**예상 총 비용**: **$100-200** (클라우드 학습 비용)
**최종 성공 확률**: **90%+** (백업 전략 및 조기 검증 체계 포함)

이러한 추가 개선사항들을 적용하면 더욱 안정적이고 실용적인 프로젝트 진행이 가능할 것입니다.
