# 악보 기호 객체 감지(Object Detection) 실행 계획

**작성일**: 2025년 7월 22일
**문서 목적**: 마디 검출 이후 단계로, 악보 내의 음표, 쉼표 등 주요 기호를 객체 감지 모델을 통해 인식하기 위한 상세 실행 계획 수립

---

## 🎯 최종 목표

- `ScoreEye` 프로젝트의 핵심 기능을 마디 인식에서 **개별 음악 기호 인식**으로 확장한다.
- 학습된 모델을 통해 악보 이미지에서 **음표, 쉼표, 음자리표 등**의 위치와 종류를 자동으로 식별한다.
- 최종적으로, 인식된 기호 정보를 조합하여 디지털 악보 데이터(예: MusicXML)로 변환할 수 있는 기반을 마련한다.

---

## 🚀 핵심 전략: 마디 단위(Measure-wise) 접근법

전체 악보를 한 번에 처리하는 대신, 이미 구현된 마디 검출 기능을 활용하여 **"보표(System) 내 마디(Measure)"**를 기본 처리 단위로 사용한다.

- **장점**:
  1.  **문제 단순화**: 복잡한 배경과 객체 수를 줄여 모델이 학습에 집중할 수 있도록 함.
  2.  **정확도 향상**: 정형화된 입력(5선 + 1마디)으로 모델 성능 극대화.
  3.  **데이터셋 구축 용이**: 작고 일관된 이미지 단위로 라벨링 작업이 효율적임.
  4.  **체계적 파이프라인**: `페이지 → 시스템 그룹 → 마디 → 기호 인식`으로 이어지는 명확한 구조.

---

## 🔧 상세 실행 파이프라인

### **Phase 1: 데이터셋 생성 (Dataset Generation)**

- **도구**: `extract_measures.py` 스크립트 활용.
- **프로세스**:
  1.  `extract_measures.py`를 대상 PDF에 실행한다.
      ```bash
      python extract_measures.py "pdfs/1-1. La Gazza ladra Overture_완판(20250202).pdf" --dpi 300
      ```
  2.  스크립트는 `output/measures/` 디렉토리 내에 페이지별 폴더를 생성한다.
  3.  각 페이지 폴더에는 다음이 포함된다:
      - **개별 마디 이미지 파일**: `[페이지]_[마디번호].png` 형식 (예: `01_001.png`)
      - **메타데이터 파일**: `metadata.json`
- **`metadata.json`의 역할**:
  - 각 마디 이미지의 **절대 위치 정보**를 담고 있는 핵심 파일.
  - 포함 정보: 페이지 번호, 페이지 내 절대 마디 번호, 시스템 그룹 정보, 원본 페이지에서의 Bounding Box 좌표, 마디 내 오선 상대 좌표 등.
  - 이 파일은 **후처리 단계에서 객체 감지 결과를 원본 악보의 의미 있는 정보로 변환**하는 데 결정적인 역할을 한다.

### **Phase 2: 데이터 라벨링 (Data Labeling)**

- **대상**: Phase 1에서 생성된 개별 마디 이미지들.
- **라벨링 도구**:
  - **Roboflow**: 웹 기반 협업 플랫폼. 데이터 증강(Augmentation), 포맷 변환(YOLO, COCO) 기능 내장. (강력 추천)
  - **LabelImg / Label Studio**: 로컬 설치형 오픈소스 도구.
- **라벨링 클래스 정의**:
  - `20250722_01_score_datasets_for_detection.md`에서 조사한 **MUSCIMA++** 데이터셋의 클래스를 기준으로 시작한다.
  - **초기 핵심 클래스**:
    - `notehead-full` (채워진 머리)
    - `notehead-half` (빈 머리)
    - `stem` (기둥)
    - `beam` (빔)
    - `quarter-rest` (4분 쉼표)
    - `half-rest` (2분 쉼표)
    - `whole-rest` (온쉼표)
    - `g-clef` (높은음자리표)
    - `f-clef` (낮은음자리표)
    - `dot` (점)
    - `sharp` (샵)
    - `flat` (플랫)
- **작업**: 각 마디 이미지 내의 모든 목표 클래스에 대해 Bounding Box를 그리고 정확한 클래스를 할당한다.

### **Phase 3: 모델 학습 (Model Training)**

- **모델 아키텍처**: **YOLO (You Only Look Once)** 계열 (예: YOLOv8)로 시작.
  - **이유**: 실시간에 가까운 빠른 속도와 높은 정확도의 균형이 잘 맞음. PyTorch 기반으로 구현이 용이함.
- **학습 전략**: **전이 학습 (Transfer Learning)**
  - COCO 데이터셋 등으로 사전 학습된(pre-trained) YOLO 모델을 기반으로, 우리가 라벨링한 악보 기호 데이터셋을 이용해 Fine-tuning 한다.
  - 이를 통해 적은 데이터로도 높은 성능을 기대할 수 있다.
- **프로세스**:
  1.  **데이터 분할**: 라벨링된 데이터셋을 Training / Validation / Test 세트로 분할 (예: 80% / 10% / 10%).
  2.  **데이터 증강 (Data Augmentation)**: 모델의 강건성을 높이기 위해 밝기 조절, 약간의 회전, 노이즈 추가 등의 기법을 적용. (Roboflow 사용 시 자동화 가능)
  3.  **학습 실행**: YOLOv8 프레임워크를 사용하여 모델 학습. `batch_size`, `learning_rate` 등 하이퍼파라미터 튜닝.
  4.  **성능 평가**: Validation set에 대한 mAP(mean Average Precision) 점수를 모니터링하며 최적의 모델 가중치(weights)를 저장.

### **Phase 4: 추론 및 후처리 (Inference & Post-processing)**

- **추론 (Inference)**:
  1.  학습된 YOLO 모델 가중치(`best.pt`)를 로드한다.
  2.  새로운 악보가 들어오면, Phase 1과 동일한 프로세스로 마디 이미지를 추출한다.
  3.  추출된 각 마디 이미지에 대해 모델을 실행하여 기호들의 Bounding Box와 클래스를 예측한다.
- **후처리 (Post-processing)**:
  1.  모델이 예측한 결과는 **마디 이미지 내의 상대 좌표**이다.
  2.  해당 마디의 `metadata.json` 파일을 읽어온다.
  3.  JSON 안의 `bounding_box_on_page` 정보를 사용하여, 예측된 기호의 상대 좌표를 **원본 페이지의 절대 좌표**로 변환한다.
  4.  `staff_line_coordinates_in_measure` 정보를 이용해 각 음표의 정확한 **음높이(pitch)**를 결정한다.
  5.  모든 정보를 종합하여 (페이지 번호, 시스템 그룹, 마디 번호, 기호 종류, 절대 위치, 음높이 등) 구조화된 데이터로 재구성한다.

---

## 📊 성공 지표

1.  **모델 성능**: Test set에 대한 **mAP@0.5 > 90%** 달성.
2.  **End-to-End 정확도**: `La Gazza ladra Overture` 1페이지의 모든 기호를 95% 이상 정확하게 인식 및 분류.
3.  **처리 속도**: 페이지당 모든 프로세스(마디 추출 ~ 기호 인식)를 10초 이내에 완료.

---

## 🗓️ 예상 마일스톤

- **Week 1**: 데이터셋 준비
  - [x] `extract_measures.py` 스크립트 완성.
  - [ ] 샘플 PDF 2~3개에 대해 마디 이미지 데이터셋 생성 완료.
  - [ ] Roboflow 프로젝트 생성 및 최소 50~100개 마디 이미지 라벨링 완료.
- **Week 2-3**: 모델 학습 및 튜닝
  - [ ] YOLOv8 학습 환경 구축.
  - [ ] 1차 모델 학습 및 성능 평가.
  - [ ] 데이터 증강 및 하이퍼파라미터 튜닝을 통한 성능 개선.
- **Week 4**: 추론 파이프라인 구축
  - [ ] 추론 및 후처리 스크립트 작성.
  - [ ] `metadata.json`을 이용한 좌표 변환 및 정보 통합 기능 구현.
  - [ ] 전체 파이프라인 통합 테스트 및 성능 검증.

---

## 💡 추가 제안사항

### 1. 점진적 클래스 확장 전략

초기 구현의 복잡도를 낮추고 빠른 성과를 위해 **단계적 클래스 확장** 접근법을 권장합니다:

- **Phase 1 (최소 기능 집합)**: 5개 클래스만으로 시작
  - `notehead-full` (채워진 음표머리)
  - `notehead-half` (빈 음표머리)
  - `quarter-rest` (4분 쉼표)
  - `half-rest` (2분 쉼표)
  - `whole-rest` (온쉼표)

- **Phase 2 (기본 기능 확장)**: 음표 구성 요소 추가
  - `stem` (기둥)
  - `beam` (빔)
  - `g-clef`, `f-clef` (음자리표)

- **Phase 3 (완전한 기능)**: 변화표 및 추가 기호
  - `dot`, `sharp`, `flat`
  - 추가 쉼표 종류
  - 다이나믹 기호 등

이 접근법의 장점:
- 초기 라벨링 작업량 감소로 빠른 프로토타입 개발 가능
- 각 단계에서 학습한 교훈을 다음 단계에 적용 가능
- 점진적인 성능 향상 측정 가능

### 2. 음높이 결정 로직 구체화

`staff_line_coordinates_in_measure` 정보를 활용한 음높이 결정 알고리즘:

```python
def determine_pitch(notehead_bbox, staff_lines):
    """
    음표머리의 bounding box와 오선 정보를 이용해 음높이 결정
    
    Args:
        notehead_bbox: 음표머리의 (x1, y1, x2, y2) 좌표
        staff_lines: 5개 오선의 y 좌표 리스트 (위에서 아래로)
    
    Returns:
        pitch: 음높이 정보 (예: 'G4', 'A4', 'B4' 등)
    """
    # 음표머리 중심의 y 좌표 계산
    notehead_center_y = (notehead_bbox[1] + notehead_bbox[3]) / 2
    
    # 오선 간격 계산
    staff_spacing = (staff_lines[4] - staff_lines[0]) / 4
    
    # 각 오선 및 간(space)에 대한 상대 위치 계산
    # 오선 위: -0.5, 오선 상: 0, 오선 아래 간: 0.5 등
    relative_position = None
    min_distance = float('inf')
    
    # 오선과 간에 대한 위치 매핑
    for i, line_y in enumerate(staff_lines):
        distance = abs(notehead_center_y - line_y)
        if distance < min_distance:
            min_distance = distance
            # 오선 상에 위치
            if distance < staff_spacing * 0.25:
                relative_position = i * 2  # 0, 2, 4, 6, 8 (오선)
            # 오선 사이 간에 위치
            elif i < 4:
                next_line_distance = abs(notehead_center_y - staff_lines[i + 1])
                if distance < next_line_distance:
                    relative_position = i * 2 + 1  # 1, 3, 5, 7 (간)
    
    # 음자리표 종류에 따른 음높이 매핑 (높은음자리표 기준 예시)
    pitch_map_treble = {
        -2: 'C6',  # 위 덧줄
        -1: 'B5',  # 위 덧간
        0: 'A5',   # 첫 번째 줄
        1: 'G5',   # 첫 번째 간
        2: 'F5',   # 두 번째 줄
        3: 'E5',   # 두 번째 간
        4: 'D5',   # 세 번째 줄
        5: 'C5',   # 세 번째 간
        6: 'B4',   # 네 번째 줄
        7: 'A4',   # 네 번째 간
        8: 'G4',   # 다섯 번째 줄
        9: 'F4',   # 아래 덧간
        10: 'E4'   # 아래 덧줄
    }
    
    return pitch_map_treble.get(relative_position, 'Unknown')
```

### 3. 데이터 검증 및 품질 관리

라벨링 데이터의 품질을 보장하기 위한 추가 단계:

- **교차 검증**: 동일한 마디를 2명이 독립적으로 라벨링하고 비교
- **자동 검증 스크립트**: 
  - 음표머리와 기둥이 연결되어 있는지 확인
  - 빔이 적절한 음표들을 연결하는지 검증
  - 오선 밖의 비정상적인 위치에 라벨링된 객체 검출
- **시각화 도구**: 라벨링 결과를 원본 이미지에 오버레이하여 빠른 검토 가능
