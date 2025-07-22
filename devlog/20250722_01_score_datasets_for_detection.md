# 공개 악보 객체 감지 데이터셋 안내

음표를 Object Detection 방식으로 감지하기 위해 사용할 수 있는 주요 공개 데이터셋들을 정리합니다.

---

## 1. MUSCIMA++ (기보 기호 감지 + 라벨링)

- 📎 URL: https://github.com/DDMAL/MUSCIMA++
- 구성:
  - 140개의 손으로 쓴 악보 이미지 (오선 제거됨)
  - 각 이미지에 대해 music symbol 객체의 bbox + class 정보
  - 라벨링: XML 형태 → JSON/YOLO 포맷으로 변환 필요
- 특징:
  - 실험적 연구에 널리 사용됨
  - 다양한 음표, 쉼표, 기호 포함
- 라벨 예시 클래스:
  - notehead-full, stem, dot, rest, clef 등

---

## 2. DeepScores (기호 검출 대형 데이터셋)

- 📎 URL: https://deepscores.org/
- 구성:
  - 수천 장의 고해상도 악보 이미지
  - 120개 이상의 기호 클래스 (악보 전반 커버)
- 특징:
  - 대규모 학습에 적합
  - 압축된 `.npz` 형식 annotation → 변환 필요
- 단점:
  - 접근성 다소 복잡, 변환 도구 필요
- 활용 시:
  - 특정 기호만 추려서 fine-tuning 용도로 사용 가능

---

## 3. OpenOMR Symbol Datasets

- 일부 GitHub 프로젝트에 YOLO/COCO 포맷으로 정리된 소형 symbol 데이터셋 존재
- 예: https://github.com/CalPolyCSC/OpenOMR

---

## 변환 팁

- MUSCIMA++ → YOLO 포맷 변환 스크립트 예제 있음 (GitHub 커뮤니티 참고)
- Roboflow에서 JSON 업로드 → 자동 변환 가능
- DeepScores는 사전 가공된 일부 subset이 GitHub에 공개된 경우도 있음

---

## 추천 순서

1. 🎯 **MUSCIMA++**로 시작 → 작고 다루기 쉬움, 라벨 양질
2. 필요 시 **DeepScores** 추가 → 대형 학습
3. YOLO 학습: Roboflow 또는 local 환경에서 빠르게 시작 가능

---

## 참고 코드 리포

- MUSCIMA++ YOLO 포맷 변환 예제: https://github.com/MonkVG/MUSCIMA-YOLO
- DeepScores 변환 예제: https://github.com/Tobias-Fischer/DeepScoresConverter