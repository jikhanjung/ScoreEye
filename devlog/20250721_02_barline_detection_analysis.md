# 바라인(Barline) 검출 문제 분석 및 개선 방안

**작성일**: 2025년 7월 21일  
**문서 목적**: 현재 바라인 검출 알고리즘의 문제점 분석 및 개선 방안 제시

---

## 🚨 현재 문제 상황

screenshots/measure.png 파일의 분석 결과, 오선(staff lines)은 정확히 검출하지만 바라인(barlines)은 전혀 검출하지 못하는 상황입니다.

### 검출 실패 사례
- **빨간 동그라미 표시 영역**: 명확한 바라인들이 존재하지만 알고리즘이 인식하지 못함
- **검출 결과**: 0 barline candidates, 0 valid barlines
- **오선 검출**: 정상 동작 (60+ staff lines detected)

---

## 🔍 문제점 상세 분석

### 1. **바라인 특성에 대한 잘못된 가정**

**현재 알고리즘의 가정**:
```python
# detect_measure.py:235 - 긴 수직 커널 사용
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
```

**실제 바라인의 특성**:
- 바라인은 **연속적인 긴 수직선이 아님**
- **오선만 교차하는 짧은 수직 세그먼트들**로 구성
- 오선 사이의 공백 부분에는 바라인이 존재하지 않음

### 2. **과도한 연결성 요구**

**문제가 되는 코드**:
```python
# detect_measure.py:383-412 - _is_continuous_barline()
# 최대 3픽셀 갭만 허용하며 연속성을 엄격하게 요구
def _is_continuous_barline(self, column, start_y, end_y, max_gap=3):
```

**실제 상황**:
- 오선 간격(약 10-15픽셀)만큼 공백이 정상적으로 존재
- 현재 알고리즘이 이를 "끊어진 선"으로 잘못 판단

### 3. **형태학적 연산의 한계**

**현재 방식의 문제**:
```python
# detect_measure.py:238 - MORPH_OPEN으로 수직선 추출
vertical_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel)
```

- 긴 커널이 짧은 바라인 세그먼트들을 제거
- 오선과 바라인의 교차점만 남기지 못함

### 4. **검증 로직의 과도한 엄격성**

**문제 코드**:
```python
# detect_measure.py:576-578 - 5개 오선 모두와 교차 요구
if intersections != 5:
    return False
```

**실제 악보에서**:
- 일부 바라인은 4개 오선만 교차할 수 있음
- 스캔 품질이나 인쇄 상태에 따라 일부 교차점이 불분명할 수 있음

---

## 💡 개선 방안

### **방안 1: 세그먼트 기반 바라인 검출**

각 오선 위치에서 개별적으로 수직 세그먼트를 검출하는 방식:

```python
def detect_barlines_by_staff_segments(self, binary_img):
    """오선별 수직 세그먼트 검출 후 x좌표 기준 클러스터링"""
    
    barline_candidates = []
    
    # 각 오선에서 수직 세그먼트 검출
    for staff_y in self.staff_lines:
        # 오선 주변 좁은 ROI 설정 (±3픽셀)
        roi_start = max(0, staff_y - 3)
        roi_end = min(binary_img.shape[0], staff_y + 4)
        roi = binary_img[roi_start:roi_end, :]
        
        # ROI에서 수직 projection
        vertical_projection = np.sum(roi, axis=0)
        
        # 임계값 이상인 x좌표들 수집
        threshold = np.max(vertical_projection) * 0.3
        candidates = np.where(vertical_projection > threshold)[0]
        
        barline_candidates.extend(candidates)
    
    # x좌표 기준으로 클러스터링하여 바라인 위치 결정
    return self._cluster_barline_candidates(barline_candidates)
```

### **방안 2: 교차점 기반 검증**

연속성 대신 교차점 개수로 바라인을 검증:

```python
def validate_barline_by_intersections(self, x):
    """오선과의 교차점 개수로 바라인 검증 (완화된 기준)"""
    
    intersections = 0
    staff_groups = self._get_staff_groups()  # 5개씩 그룹핑
    
    for staff_group in staff_groups:
        group_intersections = 0
        
        for staff_y in staff_group:
            # 오선 주변 window에서 교차점 검사
            y_start = max(0, staff_y - 2)
            y_end = min(self.binary_img.shape[0], staff_y + 3)
            x_start = max(0, x - 1)
            x_end = min(self.binary_img.shape[1], x + 2)
            
            window = self.binary_img[y_start:y_end, x_start:x_end]
            
            if np.any(window > 0):
                group_intersections += 1
        
        # 5개 오선 중 최소 4개와 교차하면 유효한 바라인으로 판정
        if group_intersections >= 4:
            return True
    
    return False
```

### **방안 3: 적응적 커널 크기**

오선 간격에 기반한 동적 커널 크기 조정:

```python
def get_adaptive_kernel_size(self):
    """오선 간격 기반 적응적 커널 크기 계산"""
    
    if len(self.staff_lines) < 2:
        return 15  # 기본값
    
    # 오선 간 평균 간격 계산
    spacings = []
    for i in range(len(self.staff_lines) - 1):
        spacing = self.staff_lines[i+1] - self.staff_lines[i]
        if spacing < 30:  # 동일 스태프 내 간격만 고려
            spacings.append(spacing)
    
    avg_spacing = np.median(spacings) if spacings else 12
    
    # 커널 높이를 오선 간격의 70%로 설정
    # 이렇게 하면 오선은 감지하되 오선 사이 공백은 무시
    return max(8, int(avg_spacing * 0.7))
```

### **방안 4: 2단계 검출 프로세스**

1단계에서 후보를 대략 검출하고, 2단계에서 정밀 검증:

```python
def detect_barlines_two_stage(self, binary_img):
    """2단계 바라인 검출 프로세스"""
    
    # 1단계: 관대한 기준으로 후보 수집
    stage1_candidates = self._collect_barline_candidates_liberal(binary_img)
    
    # 2단계: 엄격한 검증으로 필터링
    stage2_validated = []
    
    for x in stage1_candidates:
        if self._validate_barline_strict(x):
            stage2_validated.append(x)
    
    return self._merge_nearby_barlines(stage2_validated)

def _collect_barline_candidates_liberal(self, binary_img):
    """1단계: 관대한 기준으로 바라인 후보 수집"""
    candidates = []
    
    # 작은 커널로 세밀한 수직선 검출
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    vertical_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, small_kernel)
    
    # 각 열의 수직 content 평가
    for x in range(binary_img.shape[1]):
        column = vertical_lines[:, x]
        
        # 오선 영역에서의 픽셀 밀도 확인
        staff_region_pixels = self._count_pixels_in_staff_regions(column, x)
        
        if staff_region_pixels >= 3:  # 최소 3개 오선 영역에서 픽셀 존재
            candidates.append(x)
    
    return candidates
```

---

## 🎯 구현 우선순위

### **Phase 1: 즉시 적용 (High Priority)**
1. **세그먼트 기반 검출** - 가장 직접적인 해결책
2. **교차점 기준 완화** - 5개 → 4개 오선 교차 허용
3. **적응적 커널 크기** - 오선 간격 기반 동적 조정

### **Phase 2: 성능 개선 (Medium Priority)**
1. **2단계 검출 프로세스** - 정확도와 재현율 균형
2. **ROI 기반 처리** - 연산 효율성 향상
3. **다중 스태프 처리** - 복잡한 악보 대응

### **Phase 3: 고도화 (Low Priority)**
1. **머신러닝 기반 후처리** - 오탐 제거
2. **특수 바라인 처리** - 더블 바라인, 반복 기호
3. **사용자 피드백 학습** - 검출 성능 지속 개선

---

## 🧪 테스트 계획

### **Unit Tests**
- `test_staff_segment_detection()` - 개별 오선에서 세그먼트 검출
- `test_intersection_validation()` - 교차점 기반 검증 로직
- `test_adaptive_kernel()` - 동적 커널 크기 계산

### **Integration Tests**  
- `test_full_barline_detection()` - 전체 파이프라인 테스트
- `test_multi_staff_scores()` - 다중 스태프 악보 테스트
- `test_various_scan_qualities()` - 다양한 스캔 품질 대응

### **Validation Data**
- La Gazza ladra Overture (현재 실패 케이스)
- 다양한 악기 편성의 악보 샘플
- 서로 다른 인쇄 품질의 악보들

---

## 📊 예상 개선 효과

**현재 상태**:
- Barline Detection Rate: 0% (screenshots/measure.png 기준)
- Staff Line Detection: 100% (정상 동작)

**개선 후 예상**:
- Barline Detection Rate: 85-90%
- False Positive Rate: < 5%
- Processing Time: 현재 대비 20% 증가 (acceptable)

---

## 🔧 다음 단계

1. **세그먼트 기반 검출 구현** - `detect_barlines_by_staff_segments()` 함수 작성
2. **기존 코드와 A/B 테스트** - 성능 비교 검증
3. **실제 악보 데이터로 검증** - 다양한 케이스 테스트
4. **최적 파라미터 튜닝** - 임계값, 커널 크기 등 조정
5. **Production 배포** - 안정성 확인 후 메인 브랜치 적용
