# 마디 개수 자동 인식 시스템 구현 계획

본 문서는 악보 이미지에서 마디 개수를 자동으로 인식하는 시스템의 기술적 구현 계획을 정리한 것입니다. 목표는 바라인(barline)을 감지하여 마디 구간을 정의하고, 전체 마디 개수를 추정하는 것입니다.

---

## 🎯 목표

- 입력: 스캔된 악보 이미지 또는 PDF
- 출력: 인식된 마디 개수, 마디 위치 시각화(Optional)

---

## 📦 주요 기술 구성

| 단계 | 기술 | 설명 |
|------|------|------|
| 1 | 이미지 전처리 | 이진화, 노이즈 제거 |
| 2 | 오선(staff line) 감지 | Horizontal projection, peak 검출 |
| 3 | 바라인(barline) 후보 검출 | Hough Line Transform 또는 Morphological 수직선 추출 |
| 4 | 바라인 필터링 | 오선을 완전히 교차하는 수직선만 남김 |
| 5 | 바라인 정렬 및 중복 제거 | X좌표 기반 정렬 및 가까운 바라인 병합 |
| 6 | 마디 개수 계산 | 바라인 수 - 1 |
| 7 | 결과 시각화 (선택) | 이미지 위에 마디 경계 표시 또는 번호 출력 |

---

## 🔧 구현 흐름 (OpenCV 기반 예)

### 1. 이미지 로딩 및 전처리
```python
img = cv2.imread("score.png", 0)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
```

### 2. 오선 감지
```python
projection = np.sum(binary, axis=1)
peaks = find_peaks(projection)
# 오선 위치 후보 리스트 생성
```

### 3. 바라인 후보 검출
```python
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
```

또는 Hough Line Transform 사용
```python
lines = cv2.HoughLinesP(vertical_lines, 1, np.pi / 180, threshold=100, minLineLength=30, maxLineGap=5)
```

### 4. 바라인 필터링 (오선 교차 기준)
- 각 수직선이 오선 y좌표 5개를 모두 통과하는지 판단
- 바라인 후보 중 실제 바라인만 남김

### 5. 바라인 정렬 및 병합
```python
# x 좌표 기준 정렬
# 가까운 거리(예: 5px 이하) 내에 있는 바라인 병합
```

### 6. 마디 개수 계산
```python
num_barlines = len(filtered_barlines)
num_measures = num_barlines - 1
```

---

## 🧠 고려 사항

- 붙점이나 음표기둥과의 혼동 방지
- 더블 바라인 병합 필요
- 반복 기호(:||) 등은 특별 처리 고려
- 악보가 2단(피아노) 이상일 경우 줄 그룹 처리 필요

---

## 💡 확장 가능성

- 마디 단위 Crop 이미지 생성
- 마디 번호 인식 및 텍스트 출력
- 합주 악보용 마디 동기화

---

## 📁 파일 구조 예시
```
project/
├── images/
│   └── sample_score.png
├── output/
│   └── barline_overlay.png
├── detect_measure.py
└── README.md
```

---

## ✅ 요약

- 바라인을 검출하고 간격을 분석함으로써 마디 수를 비교적 정확히 산출 가능
- 오선 감지와 수직선 필터링이 정확도의 핵심
- OpenCV만으로도 실용적인 수준의 자동 마디 인식 가능