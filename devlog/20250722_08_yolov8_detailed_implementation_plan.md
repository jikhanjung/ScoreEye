### **DeepScores 기반 YOLOv8 음표 인식 모델 상세 구현 계획**

#### **🎯 최종 목표**
- DeepScores 데이터셋으로 학습된 YOLOv8 모델(`best.pt`)을 확보한다.
- 이 모델을 `ScoreEye` 프로젝트에 통합하여, `extract_measures.py`로 추출된 마디 이미지 내의 개별 악보 기호(음표, 쉼표 등)를 감지하는 기능을 구현한다.

---

### **Phase 1: 환경 설정 및 데이터 준비 (1일)**

1.  **필수 라이브러리 설치**:
    - `requirements.txt` 파일에 다음 라이브러리를 추가하고 설치합니다.
      ```
      ultralytics
      numpy
      opencv-python
      pyyaml
      tqdm
      ```
    - 설치 명령어:
      ```bash
      pip install -r requirements.txt
      ```

2.  **DeepScores 데이터셋 다운로드**:
    - 공식 웹사이트(https://deepscores.org/dataset/)에서 "DeepScoresV2 (Main dataset)"를 다운로드합니다.
    - 다운로드 받은 파일의 압축을 `dataset/deepscores_v2` 와 같은 디렉터리에 해제합니다.

3.  **프로젝트 디렉터리 구조 설정**:
    - 데이터셋 변환 및 학습을 위해 다음과 같은 구조를 프로젝트 루트에 생성합니다.
      ```
      ScoreEye/
      ├── deepscores_workspace/
      │   ├── data/                 # YOLOv8 학습용 데이터셋 최종 위치
      │   │   ├── images/
      │   │   │   ├── train/
      │   │   │   └── val/
      │   │   ├── labels/
      │   │   │   ├── train/
      │   │   │   └── val/
      │   │   └── deepscores.yaml   # 데이터셋 설정 파일
      │   ├── raw_data/             # 다운로드한 DeepScores 원본 데이터 위치
      │   └── preprocess_deepscores.py # 변환 스크립트
      ├── ... (기존 파일들)
      ```

---

### **Phase 2: 데이터 전처리 및 변환 (2-3일)**

이 단계가 가장 중요하며, `preprocess_deepscores.py` 스크립트 작성이 핵심입니다.

1.  **대상 클래스 선정 및 매핑**:
    - DeepScores는 135개의 클래스를 제공하지만, 초기 모델은 핵심 클래스에 집중합니다.
    - **초기 목표 클래스 (10개)**:
      - `noteheadFull` (채워진 음표 머리)
      - `noteheadHalf` (빈 음표 머리)
      - `noteheadWhole` (온음표 머리)
      - `stem` (기둥)
      - `beam` (빔)
      - `dot` (점)
      - `gClef` (높은음자리표)
      - `fClef` (낮은음자리표)
      - `restQuarter` (4분 쉼표)
      - `restHalf` (2분 쉼표)
    - 이 클래스들을 `0`부터 시작하는 정수 인덱스로 매핑하는 딕셔너리를 스크립트 내에 정의합니다.

2.  **`preprocess_deepscores.py` 스크립트 구현**:
    - 이 스크립트는 다음 기능을 수행해야 합니다.

    ```python
    # preprocess_deepscores.py

    import numpy as np
    import os
    import cv2
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split

    # 1. 설정 변수
    RAW_DATA_DIR = 'raw_data/ds2_dense'
    OUTPUT_DIR = 'data'
    TARGET_CLASSES = {
        'noteheadFull': 0, 'noteheadHalf': 1, 'noteheadWhole': 2,
        'stem': 3, 'beam': 4, 'dot': 5, 'gClef': 6, 'fClef': 7,
        'restQuarter': 8, 'restHalf': 9
    }
    IMAGE_SIZE = 1024 # DeepScores는 고해상도이므로 1024 권장
    TRAIN_RATIO = 0.85

    def convert_to_yolo(bbox, img_width, img_height):
        # DeepScores bbox [x_min, y_min, x_max, y_max] -> YOLO 포맷 변환
        dw = 1. / img_width
        dh = 1. / img_height
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_center = (bbox[1] + bbox[3]) / 2.0
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return (x_center * dw, y_center * dh, width * dw, height * dh)

    def process_dataset():
        # 2. 이미지와 NPZ 파일 목록 가져오기
        all_images = [f for f in os.listdir(os.path.join(RAW_DATA_DIR, 'images')) if f.endswith('.png')]
        
        # 3. Train/Validation 분할
        train_images, val_images = train_test_split(all_images, train_size=TRAIN_RATIO, random_state=42)

        # 4. 데이터셋 루프 (train, val)
        for subset, image_list in [('train', train_images), ('val', val_images)]:
            img_path = os.path.join(OUTPUT_DIR, 'images', subset)
            lbl_path = os.path.join(OUTPUT_DIR, 'labels', subset)
            os.makedirs(img_path, exist_ok=True)
            os.makedirs(lbl_path, exist_ok=True)

            for img_name in tqdm(image_list, desc=f"Processing {subset} set"):
                # 5. 원본 이미지 리사이즈 및 저장
                original_img = cv2.imread(os.path.join(RAW_DATA_DIR, 'images', img_name))
                h, w, _ = original_img.shape
                resized_img = cv2.resize(original_img, (IMAGE_SIZE, IMAGE_SIZE))
                cv2.imwrite(os.path.join(img_path, img_name), resized_img)

                # 6. NPZ 어노테이션 로드 및 파싱
                npz_path = os.path.join(RAW_DATA_DIR, 'annotations_npz', img_name.replace('.png', '.npz'))
                annotations = np.load(npz_path, allow_pickle=True)['arr_0']
                
                yolo_labels = []
                for ann in annotations:
                    class_name = ann['class_name']
                    if class_name in TARGET_CLASSES:
                        class_id = TARGET_CLASSES[class_name]
                        bbox = ann['bbox'] # [x_min, y_min, x_max, y_max]
                        
                        # 7. YOLO 포맷으로 변환
                        yolo_bbox = convert_to_yolo(bbox, w, h)
                        yolo_labels.append(f"{class_id} {' '.join(map(str, yolo_bbox))}")

                # 8. 라벨 파일 저장
                with open(os.path.join(lbl_path, img_name.replace('.png', '.txt')), 'w') as f:
                    f.write('
'.join(yolo_labels))

    if __name__ == '__main__':
        process_dataset()
        print("DeepScores dataset conversion to YOLOv8 format is complete.")

    ```

3.  **변환 스크립트 실행**:
    ```bash
    cd deepscores_workspace
    python preprocess_deepscores.py
    ```
    - 실행이 완료되면 `deepscores_workspace/data` 디렉터리에 학습 준비가 완료된 데이터가 생성됩니다.

---

### **Phase 3: 모델 학습 (2-4일)**

1.  **`deepscores.yaml` 파일 생성**:
    - `deepscores_workspace/data` 디렉터리에 다음 내용으로 파일을 생성합니다.

    ```yaml
    # deepscores.yaml
    path: /home/user/projects/ScoreEye/deepscores_workspace/data  # 절대 경로 사용 권장
    train: images/train
    val: images/val

    # Classes
    names:
      0: noteheadFull
      1: noteheadHalf
      2: noteheadWhole
      3: stem
      4: beam
      5: dot
      6: gClef
      7: fClef
      8: restQuarter
      9: restHalf
    ```

2.  **초기 테스트 학습 (Sanity Check)**:
    - 전체 파이프라인이 정상 동작하는지 확인하기 위해 적은 epoch으로 빠르게 학습을 실행합니다.
    - 프로젝트 루트 디렉터리(`ScoreEye/`)에서 실행합니다.

    ```bash
    yolo task=detect mode=train model=yolov8n.pt data=./deepscores_workspace/data/deepscores.yaml epochs=5 imgsz=1024 batch=16
    ```

3.  **본 학습 실행**:
    - 테스트 학습이 성공적으로 완료되면, 충분한 epoch으로 본 학습을 진행합니다.
    - `patience` 옵션은 val loss가 개선되지 않을 때 조기 종료하여 시간을 절약합니다.

    ```bash
    yolo task=detect mode=train model=yolov8s.pt data=./deepscores_workspace/data/deepscores.yaml epochs=100 imgsz=1024 batch=8 patience=10
    ```
    - `yolov8s.pt` (Small) 모델로 시작하는 것을 권장하며, 성능에 따라 `m` 또는 `l` 모델을 사용할 수 있습니다.
    - 학습이 완료되면 `runs/detect/train/weights/best.pt` 경로에 최적의 모델 가중치 파일이 생성됩니다.

---

### **Phase 4: 추론 및 프로젝트 통합 (2일)**

1.  **학습된 모델 이동**:
    - 생성된 `best.pt` 파일을 프로젝트의 `models/` 와 같은 관리 디렉터리로 복사하고, `deepscores_yolov8s.pt` 와 같이 이름을 변경합니다.

2.  **추론 스크립트 작성 (`symbol_detector.py`)**:
    - 이 스크립트는 마디 이미지를 입력받아 기호들을 감지하고 결과를 반환하는 역할을 합니다.

    ```python
    # symbol_detector.py
    from ultralytics import YOLO
    import cv2

    class SymbolDetector:
        def __init__(self, model_path):
            self.model = YOLO(model_path)

        def detect(self, image_path_or_np_array):
            """
            이미지에서 악보 기호를 감지합니다.
            
            Returns:
                A list of dictionaries, each containing 'box', 'class_name', 'confidence'.
            """
            results = self.model(image_path_or_np_array, verbose=False)
            
            detected_symbols = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id]
                    
                    detected_symbols.append({
                        'box': (x1, y1, x2, y2),
                        'class_name': class_name,
                        'confidence': conf
                    })
            return detected_symbols

    if __name__ == '__main__':
        # 테스트 실행
        detector = SymbolDetector('models/deepscores_yolov8s.pt')
        
        # extract_measures.py로 추출된 마디 이미지 경로
        measure_image_path = 'output/measures/00_page/00_001.png' 
        
        symbols = detector.detect(measure_image_path)
        
        image = cv2.imread(measure_image_path)
        for sym in symbols:
            x1, y1, x2, y2 = sym['box']
            label = f"{sym['class_name']} {sym['confidence']:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.imwrite('output/symbol_detection_result.png', image)
        print("Symbol detection result saved to 'output/symbol_detection_result.png'")
    ```

3.  **기존 파이프라인과 통합**:
    - `extract_measures.py` 또는 `scoreeye_gui.py`에서 마디 이미지를 생성한 후, `SymbolDetector`를 호출하여 각 마디의 기호를 인식하는 로직을 추가합니다.
    - 인식된 기호 정보(위치, 클래스)와 마디의 메타데이터(`metadata.json`)를 결합하여 MusicXML 생성 등 후속 작업을 위한 기반을 마련합니다.

---

### **🗓️ 예상 타임라인**

- **Week 1**:
  - [ ] 환경 설정 및 데이터 다운로드 (1일)
  - [ ] `preprocess_deepscores.py` 개발 및 테스트 (2-3일)
  - [ ] 데이터셋 변환 완료 및 샘플 확인 (1일)
- **Week 2**:
  - [ ] `deepscores.yaml` 작성 및 초기 테스트 학습 (1일)
  - [ ] 본 학습 실행 및 모니터링 (2-4일, GPU 성능에 따라 변동)
- **Week 3**:
  - [ ] `symbol_detector.py` 추론 스크립트 개발 (1일)
  - [ ] `ScoreEye` 프로젝트에 통합 및 테스트 (2일)
  - [ ] 결과 분석 및 개선 방향 수립 (1일)

이 계획을 따르면 체계적으로 음표 인식 모델을 개발하고 프로젝트에 성공적으로 통합할 수 있을 것입니다.

---

### **🔧 구현 계획 개선 제안**

#### **1. Phase 2 데이터 품질 검증 단계 강화**

현재 계획의 `preprocess_deepscores.py`는 단순 포맷 변환만 수행하므로, 다음 검증 단계들을 추가해야 합니다:

**데이터 무결성 검증**
```python
def validate_dataset_quality():
    """데이터셋 품질 검증 함수"""
    # 1. 이미지-라벨 매칭 확인
    missing_pairs = check_image_label_pairs()
    
    # 2. 바운딩 박스 유효성 검사
    invalid_bboxes = validate_bboxes()  # 0-1 범위, 너비/높이 > 0
    
    # 3. 클래스 분포 분석
    class_distribution = analyze_class_distribution()
    
    # 4. 이미지 품질 확인
    corrupted_images = detect_corrupted_images()
    
    # 검증 리포트 생성
    generate_validation_report(missing_pairs, invalid_bboxes, class_distribution, corrupted_images)
```

**클래스 불균형 분석 및 해결**
```python
def analyze_class_imbalance():
    """
    클래스 불균형 분석 및 해결책 제안
    - noteheadFull이 90%, dot이 1% 같은 심각한 불균형 발견
    - 소수 클래스에 대한 augmentation 증가
    - weighted loss 함수 사용 권장
    - focal loss 적용 고려
    """
    class_counts = count_annotations_per_class()
    imbalance_ratio = calculate_imbalance_ratio(class_counts)
    
    # 시각화
    plot_class_distribution(class_counts)
    
    # 해결책 제안
    balancing_strategy = recommend_balancing_strategy(imbalance_ratio)
    return balancing_strategy
```

**어노테이션 품질 샘플 검사**
```python
def visual_annotation_check():
    """
    무작위로 100개 이미지를 선택하여 
    바운딩 박스가 실제 기호와 정확히 매칭되는지 시각적 확인
    """
    sample_images = random.sample(all_images, 100)
    annotation_quality_scores = []
    
    for img in sample_images:
        quality_score = visualize_and_check_annotations(img)
        annotation_quality_scores.append(quality_score)
    
    # 수동 검토를 위한 HTML 리포트 생성
    generate_quality_check_report(sample_images, annotation_quality_scores)
```

#### **2. 모델 성능 평가 메트릭 체계 구축**

현재 계획에는 성능 평가 기준이 명시되지 않아 다음 메트릭 시스템이 필요합니다:

**기본 Detection 메트릭**
```python
class OMRMetrics:
    """OMR 특화 성능 평가 메트릭"""
    
    def __init__(self):
        self.iou_thresholds = [0.3, 0.5, 0.7]  # OMR은 0.5보다 낮은 threshold도 의미있음
        self.class_names = ['noteheadFull', 'noteheadHalf', 'stem', 'beam', 'dot', 
                           'gClef', 'fClef', 'restQuarter', 'restHalf']
        
    def evaluate_model(self, model, val_dataset):
        """종합 모델 성능 평가"""
        # 1. 표준 COCO 메트릭
        mAP_30 = self.calculate_map(iou_threshold=0.3)
        mAP_50 = self.calculate_map(iou_threshold=0.5)
        mAP_75 = self.calculate_map(iou_threshold=0.75)
        
        # 2. 클래스별 정밀도/재현율
        per_class_metrics = {}
        for class_name in self.class_names:
            precision, recall, f1 = self.calculate_precision_recall(class_name)
            per_class_metrics[class_name] = {
                'precision': precision,
                'recall': recall, 
                'f1_score': f1
            }
        
        return {
            'mAP@0.3': mAP_30,
            'mAP@0.5': mAP_50,
            'mAP@0.75': mAP_75,
            'per_class': per_class_metrics
        }
```

**OMR 전용 메트릭**
```python
def calculate_omr_specific_metrics():
    """음악 기보법 특성을 고려한 전용 메트릭"""
    
    # 1. 음표 구성요소 완성도 (Notehead + Stem 매칭률)
    note_completeness = measure_note_component_matching()
    
    # 2. 수직 정렬 정확도 (같은 시간의 음표들이 수직으로 정렬되는가)
    vertical_alignment_score = measure_vertical_alignment()
    
    # 3. 마디별 기호 밀도 분석 (과도한 false positive 검출 방지)
    symbols_per_measure = analyze_symbol_density()
    
    # 4. 클래스 혼동 매트릭스 (noteheadFull vs noteheadHalf 구분 정확도)
    confusion_matrix = calculate_class_confusion()
    
    return {
        'note_completeness': note_completeness,
        'vertical_alignment': vertical_alignment_score,
        'symbol_density': symbols_per_measure,
        'class_confusion': confusion_matrix
    }
```

**실제 사용성 메트릭**
```python
def measure_practical_performance():
    """실제 ScoreEye 파이프라인에서의 성능 측정"""
    
    pipeline_metrics = {
        'processing_time_per_measure': [],      # 마디당 처리 시간
        'memory_usage_peak': [],                # 최대 메모리 사용량
        'gpu_utilization': [],                  # GPU 활용률
        'success_rate_on_real_scores': 0.0,    # 실제 악보에서의 성공률
        'integration_compatibility': 0.0       # 기존 파이프라인과의 호환성
    }
    
    return pipeline_metrics
```

#### **3. 실패 사례 분석을 위한 시각화 도구**

디버깅과 모델 개선을 위한 체계적인 시각화 시스템:

**Detection 실패 분류 도구**
```python
class FailureAnalyzer:
    """실패 사례 자동 분류 및 분석"""
    
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset
        self.failure_categories = {
            'false_positive': [],    # 잘못 감지된 영역
            'false_negative': [],    # 놓친 기호들  
            'misclassification': [], # 잘못 분류된 기호들
            'bbox_inaccuracy': [],   # 위치는 맞지만 박스 부정확
        }
    
    def categorize_failures(self):
        """실패 사례를 유형별로 자동 분류"""
        for image, ground_truth in self.test_dataset:
            predictions = self.model(image)
            failures = self.compare_predictions_gt(predictions, ground_truth)
            self.categorize_by_type(failures)
        
        return self.generate_failure_report()
```

**Interactive 시각화 대시보드**
```python
def create_failure_dashboard():
    """
    Streamlit 또는 Flask 기반 웹 대시보드
    - 실패 사례를 카테고리별로 필터링
    - 각 실패 사례에 대한 상세 분석 (confidence score, IoU 값 등)
    - 개선 우선순위 제안
    """
    
    dashboard_features = [
        'failure_heatmap',          # 어떤 위치에서 실패가 많이 발생하는가
        'class_confusion_matrix',   # 어떤 클래스 간 혼동이 많은가  
        'confidence_distribution',  # 낮은 confidence의 정확한 detection vs 높은 confidence의 false positive
        'scale_analysis',          # 크기별 detection 성능 분석
        'context_analysis'         # 주변 기호에 따른 성능 차이
    ]
    
    return dashboard_features
```

**자동 개선 제안 시스템**
```python
def generate_improvement_recommendations():
    """실패 분석 결과 기반 자동 개선 제안"""
    
    failure_analysis = analyze_all_failures()
    
    recommendations = {
        'data_augmentation': suggest_augmentation_strategies(failure_analysis),
        'hyperparameter_tuning': suggest_parameter_changes(failure_analysis), 
        'architecture_changes': suggest_model_modifications(failure_analysis),
        'post_processing': suggest_post_processing_rules(failure_analysis)
    }
    
    # 예시: "noteheadHalf 클래스의 재현율이 65%로 낮습니다. 
    #       해당 클래스 이미지에 brightness augmentation 추가를 권장합니다."
    
    return recommendations
```

#### **4. 점진적 클래스 확장 전략**

초기 10개 클래스에서 단계적으로 확장하는 전략:

```python
expansion_phases = {
    'phase_1': {
        'classes': ['noteheadFull', 'stem', 'gClef'],  # 핵심 3개로 시작
        'target_mAP': 0.85,
        'duration': '3-4일'
    },
    'phase_2': {
        'classes': ['restQuarter', 'beam', 'dot'],     # 기본 리듬 요소 추가  
        'target_mAP': 0.80,
        'duration': '2-3일'
    },
    'phase_3': {
        'classes': ['sharp', 'flat', 'natural'],       # 임시표 추가
        'target_mAP': 0.75,
        'duration': '2일'
    },
    'phase_4': {
        'classes': ['timeSig4_4', 'keySigFlat1'],      # 조표/박자표 추가
        'target_mAP': 0.70,
        'duration': '2일'
    }
}

def progressive_training_strategy():
    """점진적 학습 전략 구현"""
    for phase_name, phase_config in expansion_phases.items():
        print(f"Starting {phase_name}: {phase_config['classes']}")
        
        # 이전 단계 가중치로 초기화
        model = load_previous_phase_weights() if phase_name != 'phase_1' else 'yolov8s.pt'
        
        # 해당 단계 클래스만으로 학습
        train_model(model, phase_config['classes'], phase_config['target_mAP'])
```

#### **5. 도메인 적응 전략**

DeepScores는 합성 데이터이므로 실제 스캔 악보와의 도메인 차이 해결:

```python
def domain_adaptation_pipeline():
    """실제 악보 이미지에 대한 도메인 적응"""
    
    adaptation_stages = [
        {
            'name': 'synthetic_pretraining',
            'data': 'DeepScores 합성 데이터',
            'epochs': 50,
            'description': '기본 패턴 학습'
        },
        {
            'name': 'noise_simulation',  
            'data': 'DeepScores + 노이즈/기울어짐 시뮬레이션',
            'epochs': 20,
            'description': '실제 스캔 환경 시뮬레이션'
        },
        {
            'name': 'real_data_finetuning',
            'data': 'ScoreEye 실제 처리 이미지 (수동 라벨링)',
            'epochs': 10,
            'description': '실제 데이터 미세조정'
        }
    ]
    
    # 각 단계별 순차 학습
    for stage in adaptation_stages:
        fine_tune_model(stage)
```

#### **6. 메모리 최적화 방안**

1024px 이미지 처리를 위한 메모리 효율성 개선:

```python
optimization_strategies = {
    'multi_scale_training': {
        'strategy': 'YOLOv8s with 640px 기본 학습 → 1024px fine-tuning',
        'memory_savings': '60%',
        'performance_impact': '최소 (<5%)'
    },
    'gradient_accumulation': {
        'strategy': 'batch_size=4 × accumulation=4 = effective_batch_size=16',
        'memory_savings': '75%',
        'training_time_impact': '+10%'
    },
    'mixed_precision': {
        'strategy': 'FP16 training',
        'memory_savings': '50%',
        'speedup': '2x'
    },
    'tiled_inference': {
        'strategy': '큰 이미지를 겹치는 타일로 분할하여 처리',
        'memory_savings': '80%',
        'accuracy_impact': '최소 (overlap 처리 시)'
    }
}

def implement_memory_optimization():
    """메모리 최적화 구현"""
    
    # 1. 동적 배치 크기 조정
    def adjust_batch_size_based_on_gpu():
        gpu_memory = get_gpu_memory()
        if gpu_memory < 8:  # 8GB 미만
            return 4
        elif gpu_memory < 16:  # 16GB 미만
            return 8
        else:
            return 16
    
    # 2. 점진적 이미지 크기 증가
    def progressive_image_scaling():
        training_phases = [
            {'epochs': 30, 'img_size': 640},
            {'epochs': 20, 'img_size': 832}, 
            {'epochs': 10, 'img_size': 1024}
        ]
        return training_phases
```

#### **7. 성능 모니터링 및 추적**

```python
def create_comprehensive_monitoring():
    """포괄적인 성능 모니터링 시스템"""
    
    # Weights & Biases 연동
    wandb.init(project="scoreeye-yolov8", 
               config={
                   "dataset": "DeepScores-v2-subset",
                   "model": "YOLOv8s",
                   "classes": 10
               })
    
    # 실시간 모니터링 메트릭
    monitoring_metrics = [
        'training_loss',
        'validation_mAP', 
        'per_class_precision',
        'per_class_recall',
        'gpu_memory_usage',
        'training_speed',
        'sample_predictions'  # 매 epoch 샘플 이미지 결과
    ]
    
    # 조기 종료 조건
    early_stopping_config = {
        'monitor': 'val_mAP',
        'patience': 15,
        'min_delta': 0.001,
        'restore_best_weights': True
    }
```

#### **8. 수정된 타임라인**

개선 제안을 반영한 현실적인 타임라인:

- **Week 1-2**:
  - [ ] 환경 설정 및 DeepScores 데이터 다운로드 (2일)
  - [ ] **데이터 품질 검증 시스템 구축** (2일) ⭐ 신규
  - [ ] `preprocess_deepscores.py` 개발 및 테스트 (3일)
  - [ ] **클래스 불균형 분석 및 해결방안 수립** (1일) ⭐ 신규

- **Week 3-4**:
  - [ ] **성능 평가 메트릭 시스템 구현** (2일) ⭐ 신규
  - [ ] Phase 1 클래스 (3개) 학습 및 평가 (3일)
  - [ ] **실패 사례 분석 도구 구축** (2일) ⭐ 신규
  - [ ] Phase 2 클래스 확장 및 학습 (3일)

- **Week 5-6**:
  - [ ] **도메인 적응 파이프라인 구현** (3일) ⭐ 신규
  - [ ] `symbol_detector.py` 최적화 버전 개발 (2일)
  - [ ] ScoreEye 프로젝트 통합 및 종합 테스트 (3일)
  - [ ] **성능 모니터링 대시보드 완성** (2일) ⭐ 신규

**예상 총 소요 기간**: 6주 (기존 3주 → 6주로 현실적 조정)

이러한 개선사항들이 추가되면 원래 계획의 성공률과 실용성이 크게 향상되며, 실제 프로덕션 환경에서 안정적으로 동작하는 모델을 구축할 수 있을 것입니다.
