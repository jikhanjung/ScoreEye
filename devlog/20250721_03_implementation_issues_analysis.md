# 세그먼트 기반 바라인 검출 구현 문제 분석

**작성일**: 2025년 7월 21일  
**문서 목적**: 새로 구현된 세그먼트 기반 바라인 검출 알고리즘의 문제점 분석 및 즉시 적용 가능한 수정사항 제시

---

## 🚨 현재 상황

20250721_02 문서의 개선 방안에 따라 세그먼트 기반 바라인 검출을 구현했으나, **여전히 바라인 검출율이 0%** 상태입니다.

### 구현된 주요 변경사항
- ✅ 세그먼트 기반 검출 방식 적용 (`detect_barlines()`)
- ✅ 적응적 커널 크기 계산 (`get_adaptive_kernel_size()`)
- ✅ 클러스터링 기반 바라인 위치 결정 (`_cluster_barline_candidates()`)
- ✅ 교차점 검증 기준 완화 (5개 → 4개)

---

## 🔍 구현 문제점 상세 분석

### **문제 1: 과도한 클러스터링 요구사항**

**문제가 되는 코드** (`detect_measure.py:316`):
```python
if len(current_cluster) >= 2:  # Must appear in at least 2 staff lines
    center = int(np.mean(current_cluster))
    clusters.append(center)
```

**문제점**:
- 바라인이 모든 오선에서 동일한 강도로 검출되지 않을 수 있음
- 스캔 품질, 인쇄 상태에 따라 일부 오선에서만 약하게 나타날 수 있음
- 최소 2개 스태프 라인에서 검출되어야 한다는 조건이 너무 엄격

**즉시 수정 방안**:
```python
if len(current_cluster) >= 1:  # 단일 스태프 검출도 허용
    center = int(np.mean(current_cluster))
    clusters.append(center)
```

### **문제 2: 임계값 설정이 너무 보수적**

**문제가 되는 코드** (`detect_measure.py:271`):
```python
threshold = max(0.2, np.mean(normalized[normalized > 0]) * 0.5)
```

**문제점**:
- 하한선 0.2가 너무 높음 (20% 이상만 검출)
- 동적 임계값도 평균의 50%로 보수적
- 미약한 바라인 신호도 놓치게 됨

**즉시 수정 방안**:
```python
threshold = max(0.05, np.mean(normalized[normalized > 0]) * 0.2)  # 대폭 완화
```

### **문제 3: ROI 크기가 부족**

**문제가 되는 코드** (`detect_measure.py:258-259`):
```python
roi_start = max(0, staff_y - 3)
roi_end = min(binary_img.shape[0], staff_y + 4)  # 총 7픽셀
```

**문제점**:
- 바라인이 오선보다 약간 두꺼울 수 있음
- 스캔 해상도나 인쇄 품질에 따라 바라인 폭이 달라질 수 있음
- 7픽셀 ROI가 바라인의 전체 영역을 포착하지 못할 가능성

**즉시 수정 방안**:
```python
roi_start = max(0, staff_y - 5)
roi_end = min(binary_img.shape[0], staff_y + 6)  # 11픽셀로 확장
```

### **문제 4: 디버깅 정보 부족**

**현재 상황**:
- 중간 과정에서 발견되는 후보의 수가 출력되지 않음
- 각 오선별 projection 최대값이나 임계값 정보가 없음
- 어느 단계에서 실패하는지 파악이 어려움

**필요한 디버그 출력**:
```python
print(f"Staff line {staff_y}: max_projection={np.max(vertical_projection)}, "
      f"threshold={threshold}, candidates_found={len(candidates)}")
```

### **문제 5: 전처리 단계에서의 정보 손실 가능성**

**의심되는 부분**:
- `preprocess_image()`에서 Otsu 이진화가 바라인 정보를 손실시킬 수 있음
- Gaussian blur가 얇은 바라인을 흐리게 만들 수 있음

**검증 필요사항**:
- 원본 vs 전처리된 이미지에서 바라인 영역 픽셀값 비교
- 다른 이진화 방법 (고정 임계값, 적응적 임계값) 실험

---

## 🚀 즉시 적용 가능한 수정사항

### **수정 1: 관대한 클러스터링** 
```python
# detect_measure.py:316 수정
if len(current_cluster) >= 1:  # 1개 스태프에서도 허용
```

### **수정 2: 임계값 대폭 완화**
```python
# detect_measure.py:271 수정  
threshold = max(0.05, np.mean(normalized[normalized > 0]) * 0.2)
```

### **수정 3: ROI 확장**
```python
# detect_measure.py:258-259 수정
roi_start = max(0, staff_y - 5)
roi_end = min(binary_img.shape[0], staff_y + 6)
```

### **수정 4: 디버그 출력 추가**
```python
# detect_measure.py:275 이후에 추가
if self.debug:
    print(f"Staff {staff_y}: max_proj={np.max(vertical_projection):.3f}, "
          f"threshold={threshold:.3f}, candidates={len(candidates)}")
```

### **수정 5: 전처리 실험**
```python
# preprocess_image() 메서드에 대체 이진화 옵션 추가
def preprocess_image_alternative(self, img):
    """Alternative preprocessing with fixed threshold"""
    # Skip Gaussian blur for thin lines
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    return binary
```

---

## 🧪 단계적 디버깅 계획

### **Phase 1: 파라미터 조정 (즉시 실행)**
1. 임계값을 0.05로 낮추기
2. ROI를 11픽셀로 확장
3. 클러스터링 조건을 1개로 완화
4. 디버그 출력으로 중간 결과 확인

### **Phase 2: 전처리 실험 (1차 수정 후)**
1. Gaussian blur 제거 실험
2. 고정 임계값 이진화 실험  
3. 적응적 임계값 실험
4. 원본 vs 전처리 이미지 픽셀값 비교

### **Phase 3: 알고리즘 대안 (2차 수정 후)**
1. Hough Line Transform 기반 검출
2. Template matching 방식
3. Contour 기반 바라인 검출
4. 수직 gradient 기반 검출

---

## 🎯 예상 수정 효과

### **현재 상태**
```
Staff line detection: ✅ 정상 (60+ lines detected)  
Barline candidates: ❌ 0개 검출
Clustered barlines: ❌ 0개 검출  
Final measures: ❌ 0개
```

### **Phase 1 수정 후 예상**
```
Barline candidates: 🔄 10-50개 검출 예상
Clustered barlines: 🔄 2-8개 검출 예상
Detection rate: 🔄 30-60% 예상
```

### **Phase 2 완료 후 목표**
```
Detection rate: 🎯 70-85%
False positive rate: 🎯 < 10%
Processing time: 🎯 현재 대비 +20% 이내
```

---

## 📋 체크리스트

### **즉시 실행 (Priority 1)**
- [ ] 임계값 0.2 → 0.05로 변경
- [ ] ROI 7픽셀 → 11픽셀로 확장  
- [ ] 클러스터링 조건 2개 → 1개로 완화
- [ ] 디버그 출력 추가

### **1차 검증 후 실행 (Priority 2)**  
- [ ] 전처리 방법 실험
- [ ] Gaussian blur 제거 테스트
- [ ] 고정 임계값 이진화 테스트

### **2차 검증 후 실행 (Priority 3)**
- [ ] 대안 알고리즘 구현
- [ ] 성능 비교 테스트
- [ ] 최적 파라미터 튜닝

---

## 🔧 다음 단계

1. **즉시 수정 적용** - 위의 4개 수정사항을 코드에 반영
2. **테스트 실행** - La Gazza ladra Overture로 결과 확인  
3. **디버그 로그 분석** - 중간 과정 출력으로 병목점 파악
4. **점진적 개선** - 결과에 따라 추가 파라미터 조정
5. **대안 방법 준비** - 여전히 실패시 Phase 3 알고리즘 적용

---

## 💡 핵심 가설

**현재 실패의 주요 원인**: 임계값이 너무 높아서 바라인 후보가 애초에 수집되지 않고 있음

**검증 방법**: 임계값을 0.05로 낮춘 후 디버그 출력으로 각 스태프별 후보 개수 확인

**성공 지표**: 각 스태프에서 최소 5-10개의 후보가 발견되어야 함