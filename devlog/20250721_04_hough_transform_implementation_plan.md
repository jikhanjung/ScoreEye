# HoughLinesP 기반 바라인 검출 구현 계획

**작성일**: 2025년 7월 21일  
**문서 목적**: HoughLinesP를 활용한 바라인 검출 시스템의 상세 구현 계획  
**우선순위**: #1 (가장 유망한 접근법)

---

## 🎯 전략 개요

기존의 세그먼트 기반 접근법을 포기하고, **HoughLinesP (Probabilistic Hough Line Transform)**를 활용하여 바라인을 직접 선분으로 검출하는 방식으로 전환합니다.

### **핵심 아이디어**
- 바라인을 "단절된 세그먼트들의 집합"이 아닌 **"짧은 수직 선분들의 집합"**으로 접근
- 관대한 파라미터로 모든 수직 선분을 검출한 후, 사후 필터링으로 바라인만 추출
- 각 검출된 선분의 위치, 각도, 길이를 종합적으로 분석

---

## 🔧 상세 구현 계획

### **Phase 1: 기본 HoughLinesP 검출**

#### **1.1 전처리 최적화**
```python
def preprocess_for_hough(img):
    """HoughLinesP에 최적화된 전처리"""
    
    # 1. 노이즈 제거 (약한 블러링)
    denoised = cv2.medianBlur(img, 3)  # Median 필터로 점 노이즈 제거
    
    # 2. 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 3. 적응적 이진화 (지역별 최적화)
    binary = cv2.adaptiveThreshold(enhanced, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 
                                  blockSize=15, C=10)
    
    # 4. 형태학적 정리 (작은 노이즈 제거)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return cleaned
```

#### **1.2 관대한 HoughLinesP 검출**
```python
def detect_all_vertical_lines(binary_img):
    """모든 수직에 가까운 선분 검출"""
    
    # 매우 관대한 파라미터로 시작
    lines = cv2.HoughLinesP(
        binary_img,
        rho=1,                    # 거리 해상도 (픽셀 단위)
        theta=np.pi/180,          # 각도 해상도 (1도)
        threshold=8,              # 매우 낮은 임계값 (8개 점만 있어도 선분 인정)
        minLineLength=5,          # 최소 5픽셀 길이
        maxLineGap=3              # 최대 3픽셀 갭 허용
    )
    
    return lines if lines is not None else []
```

#### **1.3 수직성 필터링**
```python
def filter_vertical_lines(lines, angle_tolerance=15):
    """수직에 가까운 선분만 필터링"""
    
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # 각도 계산 (수직선은 90도 또는 -90도)
        if x2 == x1:  # 완전 수직선
            angle = 90
        else:
            angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
            angle = abs(angle)
        
        # 수직에 가까운 선분만 선택 (90도 ± tolerance)
        if angle >= (90 - angle_tolerance):
            vertical_lines.append({
                'line': line[0],
                'center_x': (x1 + x2) // 2,
                'center_y': (y1 + y2) // 2,
                'length': np.sqrt((x2-x1)**2 + (y2-y1)**2),
                'angle': 90 if x2 == x1 else np.arctan((y2-y1)/(x2-x1)) * 180/np.pi
            })
    
    return vertical_lines
```

### **Phase 2: 바라인 후보 클러스터링**

#### **2.1 X좌표 기반 그룹핑**
```python
def group_lines_by_x_coordinate(vertical_lines, x_tolerance=8):
    """X좌표가 비슷한 선분들을 그룹화"""
    
    if not vertical_lines:
        return []
    
    # X좌표로 정렬
    sorted_lines = sorted(vertical_lines, key=lambda l: l['center_x'])
    
    groups = []
    current_group = [sorted_lines[0]]
    
    for line in sorted_lines[1:]:
        # 현재 그룹의 평균 X좌표와 비교
        group_avg_x = np.mean([l['center_x'] for l in current_group])
        
        if abs(line['center_x'] - group_avg_x) <= x_tolerance:
            current_group.append(line)
        else:
            # 새 그룹 시작
            groups.append(current_group)
            current_group = [line]
    
    # 마지막 그룹 추가
    if current_group:
        groups.append(current_group)
    
    return groups
```

#### **2.2 그룹 내 선분 분석**
```python
def analyze_line_group(group):
    """선분 그룹을 분석하여 바라인 후보인지 판단"""
    
    analysis = {
        'center_x': np.mean([l['center_x'] for l in group]),
        'x_std': np.std([l['center_x'] for l in group]),
        'total_length': sum([l['length'] for l in group]),
        'line_count': len(group),
        'y_coverage': max([l['center_y'] for l in group]) - min([l['center_y'] for l in group]),
        'avg_angle': np.mean([l['angle'] for l in group]),
        'angle_consistency': np.std([l['angle'] for l in group])
    }
    
    # 바라인 가능성 점수 계산
    score = calculate_barline_score(analysis)
    analysis['barline_score'] = score
    
    return analysis

def calculate_barline_score(analysis):
    """바라인 가능성 점수 계산 (0-100)"""
    score = 0
    
    # 1. 수직 정렬 점수 (X좌표 표준편차가 작을수록 높음)
    if analysis['x_std'] < 2:
        score += 30
    elif analysis['x_std'] < 5:
        score += 20
    elif analysis['x_std'] < 10:
        score += 10
    
    # 2. 선분 개수 점수 (많을수록 높음, 단 과도하면 감점)
    line_count = analysis['line_count']
    if 3 <= line_count <= 8:
        score += 25
    elif line_count >= 2:
        score += 15
    
    # 3. Y축 커버리지 점수 (스태프 영역을 잘 커버할수록 높음)
    if analysis['y_coverage'] > 40:
        score += 25
    elif analysis['y_coverage'] > 20:
        score += 15
    
    # 4. 각도 일관성 점수 (모든 선분이 비슷한 각도일수록 높음)
    if analysis['angle_consistency'] < 5:
        score += 20
    elif analysis['angle_consistency'] < 10:
        score += 10
    
    return min(score, 100)
```

### **Phase 3: 스태프 기반 검증**

#### **3.1 스태프 영역과의 교차 검증**
```python
def validate_barline_with_staff(barline_analysis, staff_lines):
    """스태프 라인과의 교차를 확인하여 바라인 검증"""
    
    center_x = int(barline_analysis['center_x'])
    intersections = []
    
    for staff_y in staff_lines:
        # 바라인 X좌표에서 스태프 라인 주변 확인
        intersection_found = check_intersection_at_staff(center_x, staff_y)
        if intersection_found:
            intersections.append(staff_y)
    
    # 교차점 분석
    validation_result = {
        'intersection_count': len(intersections),
        'staff_coverage_ratio': len(intersections) / len(staff_lines) if staff_lines else 0,
        'intersections': intersections,
        'is_valid_barline': len(intersections) >= 3  # 최소 3개 스태프와 교차
    }
    
    return validation_result

def check_intersection_at_staff(x, staff_y):
    """특정 X좌표에서 스태프 라인과의 교차점 확인"""
    # 스태프 라인 주변 ±3픽셀 영역에서 수직 픽셀 존재 확인
    roi_start = max(0, staff_y - 3)
    roi_end = min(binary_img.shape[0], staff_y + 4)
    
    if x < binary_img.shape[1]:
        roi_column = binary_img[roi_start:roi_end, x]
        return np.any(roi_column > 0)
    
    return False
```

#### **3.2 최종 바라인 선별**
```python
def select_final_barlines(analyzed_groups, staff_lines, min_score=40):
    """최종 바라인 선별"""
    
    final_barlines = []
    
    for group_analysis in analyzed_groups:
        # 1. 점수 기준 1차 필터링
        if group_analysis['barline_score'] < min_score:
            continue
        
        # 2. 스태프 교차 검증
        validation = validate_barline_with_staff(group_analysis, staff_lines)
        if not validation['is_valid_barline']:
            continue
        
        # 3. 최종 바라인으로 선택
        barline = {
            'x': int(group_analysis['center_x']),
            'score': group_analysis['barline_score'],
            'staff_intersections': validation['intersection_count'],
            'coverage_ratio': validation['staff_coverage_ratio']
        }
        
        final_barlines.append(barline)
    
    # X좌표 기준 정렬
    final_barlines.sort(key=lambda b: b['x'])
    
    return final_barlines
```

### **Phase 4: 통합 및 최적화**

#### **4.1 메인 검출 함수**
```python
def detect_barlines_hough(self, binary_img):
    """HoughLinesP 기반 바라인 검출 메인 함수"""
    
    # 1. 전처리 최적화
    processed_img = preprocess_for_hough(binary_img)
    
    # 2. 모든 수직 선분 검출
    all_lines = detect_all_vertical_lines(processed_img)
    if self.debug:
        print(f"Raw HoughLinesP detected: {len(all_lines)} lines")
    
    # 3. 수직성 필터링
    vertical_lines = filter_vertical_lines(all_lines, angle_tolerance=20)
    if self.debug:
        print(f"Vertical lines filtered: {len(vertical_lines)}")
    
    # 4. X좌표 기반 그룹핑
    line_groups = group_lines_by_x_coordinate(vertical_lines, x_tolerance=10)
    if self.debug:
        print(f"Line groups formed: {len(line_groups)}")
    
    # 5. 각 그룹 분석
    analyzed_groups = [analyze_line_group(group) for group in line_groups]
    
    # 6. 스태프 기반 검증 및 최종 선별
    final_barlines = select_final_barlines(analyzed_groups, self.staff_lines, min_score=30)
    
    if self.debug:
        print(f"Final barlines selected: {len(final_barlines)}")
        for i, barline in enumerate(final_barlines):
            print(f"  Barline {i+1}: x={barline['x']}, score={barline['score']:.1f}, "
                  f"intersections={barline['staff_intersections']}")
    
    return [b['x'] for b in final_barlines]
```

#### **4.2 파라미터 자동 튜닝**
```python
def auto_tune_hough_parameters(self, binary_img):
    """이미지 특성에 따른 파라미터 자동 조정"""
    
    # 이미지 크기 분석
    height, width = binary_img.shape
    pixel_density = np.sum(binary_img > 0) / (height * width)
    
    # 스태프 간격 분석
    if len(self.staff_lines) >= 2:
        avg_staff_spacing = np.median([self.staff_lines[i+1] - self.staff_lines[i] 
                                      for i in range(len(self.staff_lines)-1)])
    else:
        avg_staff_spacing = 12  # 기본값
    
    # 동적 파라미터 계산
    params = {
        'threshold': max(5, int(10 * pixel_density)),
        'minLineLength': max(3, int(avg_staff_spacing * 0.3)),
        'maxLineGap': max(2, int(avg_staff_spacing * 0.2)),
        'x_tolerance': max(5, int(width * 0.005)),  # 이미지 너비의 0.5%
        'angle_tolerance': 25 if pixel_density < 0.1 else 15  # 노이즈 많으면 관대하게
    }
    
    if self.debug:
        print(f"Auto-tuned parameters: {params}")
    
    return params
```

---

## 📊 성능 예상 및 검증 계획

### **예상 성능**
- **검출율**: 85-95% (현재 0% 대비 대폭 개선)
- **정확도**: 90-95% (스태프 교차 검증으로 높은 정밀도)
- **처리속도**: 현재 대비 1.5-2배 느림 (HoughLinesP 연산 비용)
- **강건성**: 다양한 스캔 품질에 대해 높은 안정성

### **단계별 검증 방법**

#### **Phase 1 검증**
```python
def test_basic_hough_detection():
    # 간단한 수직선 이미지로 기본 동작 확인
    test_img = create_test_image_with_vertical_lines()
    lines = detect_all_vertical_lines(test_img)
    assert len(lines) >= 3, "기본 수직선 검출 실패"
```

#### **Phase 2 검증**
```python
def test_line_grouping():
    # 근접한 선분들이 올바르게 그룹핑되는지 확인
    test_lines = create_test_vertical_lines()
    groups = group_lines_by_x_coordinate(test_lines)
    assert len(groups) == expected_group_count, "그룹핑 로직 오류"
```

#### **Phase 3 검증**
```python
def test_staff_intersection():
    # 실제 악보 이미지에서 스태프 교차 검증
    real_score_image = load_test_score()
    # ... 검증 로직
```

---

## 🔧 구현 순서 및 마일스톤

### **Week 1: 기본 구현**
- [ ] `preprocess_for_hough()` 구현
- [ ] `detect_all_vertical_lines()` 구현  
- [ ] `filter_vertical_lines()` 구현
- [ ] 기본 검출 테스트

### **Week 2: 고급 분석**
- [ ] `group_lines_by_x_coordinate()` 구현
- [ ] `analyze_line_group()` 구현
- [ ] `calculate_barline_score()` 구현
- [ ] 그룹 분석 테스트

### **Week 3: 검증 및 통합**
- [ ] `validate_barline_with_staff()` 구현
- [ ] `select_final_barlines()` 구현
- [ ] `detect_barlines_hough()` 메인 함수 통합
- [ ] La Gazza ladra Overture 테스트

### **Week 4: 최적화 및 배포**
- [ ] `auto_tune_hough_parameters()` 구현
- [ ] 성능 최적화
- [ ] 다양한 악보 테스트
- [ ] Production 배포

---

## 🚨 위험 요소 및 대응책

### **위험 요소 1: HoughLinesP 과검출**
- **증상**: 너무 많은 선분 검출로 성능 저하
- **대응**: 적응적 임계값과 사전 필터링 강화

### **위험 요소 2: 파라미터 민감도**
- **증상**: 악보마다 다른 파라미터 최적값
- **대응**: 자동 튜닝 시스템과 robust한 기본값

### **위험 요소 3: 메모리 사용량 증가**
- **증상**: 많은 선분 데이터로 메모리 부족
- **대응**: 점진적 처리와 메모리 최적화

---

## 💡 추가 개선 아이디어

### **A급 개선사항**
1. **다중 스케일 검출**: 여러 해상도에서 검출 후 통합
2. **각도별 특화 검출**: 수직, 대각선 바라인 별도 처리
3. **템플릿 매칭 결합**: HoughLinesP + 템플릿 매칭 하이브리드

### **B급 개선사항**  
1. **기계학습 후처리**: CNN으로 바라인/비바라인 분류
2. **사용자 피드백 학습**: 검출 결과에 대한 사용자 보정 학습
3. **실시간 파라미터 조정**: 검출 과정에서 동적 파라미터 최적화

---

이 계획은 현재의 0% 검출율을 85% 이상으로 끌어올릴 수 있는 가장 현실적이고 검증된 접근법입니다.