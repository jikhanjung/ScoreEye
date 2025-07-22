# DeepScores 데이터셋을 이용한 YOLOv8 음표 인식 모델 학습 계획

본 문서는 DeepScores 공개 데이터셋을 사용하여 음악 기호(음표, 붙점, 쉼표 등)를 감지하기 위한 YOLOv8 모델 학습 계획을 정리한 것입니다.

---

## 🎯 목표

- DeepScores 데이터셋을 YOLOv8 포맷으로 전처리
- 주요 기호 클래스를 선정하여 Object Detection 모델 학습
- 학습된 모델로 악보 이미지 내 기호 위치 및 클래스 자동 감지

---

## 📦 사용 기술 및 도구

| 구성 요소 | 설명 |
|------------|------|
| DeepScores | 고해상도 악보 기호 인식 데이터셋 (.npz format) |
| YOLOv8     | Ultralytics YOLO 모델 (PyTorch 기반) |
| Python     | 전처리, 변환, 학습 자동화 |
| Roboflow (선택) | 시각적 라벨링 확인, 증강 등 |

---

## 1. 데이터셋 준비

- 📥 DeepScores 공식 사이트: https://deepscores.org
- 압축된 `.npz` annotation 포맷 포함 (클래스, 위치 정보)
- 이미지와 라벨을 추출하고 YOLOv8 형식으로 변환 필요

---

## 2. 전처리 및 변환

### 2.1 대상 클래스 선택
- 예시 대상 클래스:
  - `notehead-full`, `notehead-empty`, `dot`, `clef`, `rest`, `barline`

### 2.2 YOLO 포맷 변환
- YOLO 포맷: `class x_center y_center width height` (값은 모두 0~1 정규화)
- 변환 스크립트: Python으로 `.npz` → `.txt` 라벨 파일 생성
- 클래스 인덱스 매핑 테이블 생성 필요

### 2.3 디렉터리 구조
```
dataset/
├── images/
│   ├── train/
│   ├── val/
├── labels/
│   ├── train/
│   ├── val/
```

---

## 3. 학습 설정

### 3.1 `data.yaml` 구성
```yaml
path: dataset
train: images/train
val: images/val
names: ["notehead-full", "notehead-empty", "dot", "clef", "rest", "barline"]
```

### 3.2 학습 명령어
```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=100 imgsz=640
```

옵션:
- `yolov8n.pt`: 경량화 모델 (n=Nano, s=Small, m=Medium, l=Large)
- `imgsz`: DeepScores는 고해상도이므로 1024도 고려 가능

---

## 4. 평가 및 시각화

- `runs/detect/train` 아래에 학습 로그 및 결과 저장
- `val_batch*.jpg`, `results.png` 등으로 확인 가능
- `yolo predict` 명령으로 실제 이미지 추론

---

## 5. 후처리 및 활용

- 추론된 바운딩 박스를 바탕으로:
  - 마디 단위 grouping
  - 붙점 자동 연결
  - pitch 계산 및 MusicXML 생성

---

## 6. 향후 계획

- MUSCIMA++ 등 다른 데이터셋과의 cross-validation
- 붙점, 기둥(stem), beam 등 복잡한 기호 추가 인식 확장
- SAM2 등의 segmentation 기반 모델과 결합 고려

---

## 참고 리포지터리

- DeepScores 변환기: https://github.com/Tobias-Fischer/DeepScoresConverter
- YOLOv8 공식문서: https://docs.ultralytics.com
- MUSCIMA++: https://github.com/DDMAL/MUSCIMA++