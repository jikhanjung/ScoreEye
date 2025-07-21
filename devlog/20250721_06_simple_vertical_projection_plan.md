# 단순 수직 투영법 바라인 검출 계획

**작성일**: 2025년 7월 21일  
**문서 목적**: 가장 단순하고 관대한 방법으로 모든 수직 요소를 검출하는 시스템 설계  
**우선순위**: #3 (최후의 수단, 하지만 가장 확실한 방법)

---

## 🎯 전략 개요

복잡한 알고리즘이 모두 실패했을 때를 대비한 **최대한 단순하고 관대한 접근법**입니다. "일단 수직으로 보이는 것은 모두 찾고, 나중에 걸러내자"는 철학입니다.

### **핵심 아이디어**
- 모든 복잡한 검증 로직 제거
- 단순 수직 투영으로 "뭔가 세로로 있는" 모든 위치 검출
- 최소한의 후처리로 명백히 아닌 것만 제거
- **확실히 놓치는 것보다는 잘못 잡는 것이 낫다**는 관점

---

## 🔧 상세 구현 계획

### **Phase 1: 극단적으로 관대한 전처리**

#### **1.1 최소한의 전처리**
```python
def preprocess_minimal(self, img):
    """최소한의 전처리로 정보 손실 방지"""
    
    # 1. 노이즈 제거는 최소화 (바라인도 얇아서 노이즈로 제거될 수 있음)
    # median filter만 약하게 적용
    denoised = cv2.medianBlur(img, 3)
    
    # 2. 매우 관대한 이진화
    # Otsu 대신 낮은 고정 임계값 사용
    _, binary = cv2.threshold(denoised, 180, 255, cv2.THRESH_BINARY_INV)
    
    # 3. 형태학적 정리도 최소화 (1x1 커널로만)
    kernel = np.ones((1,1), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    if self.debug:
        cv2.imshow("Minimal Preprocessing", cv2.resize(cleaned, None, fx=0.3, fy=0.3))
        cv2.waitKey(0)
    
    return cleaned
```

#### **1.2 다중 임계값 시도**
```python
def try_multiple_thresholds(self, img):
    """여러 임계값으로 이진화한 후 합치기"""
    
    # 여러 임계값으로 이진화
    thresholds = [160, 180, 200, 220]
    binary_images = []
    
    for threshold in thresholds:
        _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
        binary_images.append(binary)
        
        if self.debug:
            print(f"Threshold {threshold}: {np.sum(binary > 0)} black pixels")
    
    # 모든 이진화 결과를 OR 연산으로 합치기
    combined = np.zeros_like(img)
    for binary in binary_images:
        combined = cv2.bitwise_or(combined, binary)
    
    if self.debug:
        cv2.imshow("Combined Thresholds", cv2.resize(combined, None, fx=0.3, fy=0.3))
        cv2.waitKey(0)
    
    return combined
```

### **Phase 2: 무차별 수직 투영 검출**

#### **2.1 전체 이미지 수직 투영**
```python
def detect_all_vertical_content(self, binary_img):
    """이미지 전체에서 수직 투영으로 모든 수직 요소 검출"""
    
    height, width = binary_img.shape
    
    # 전체 이미지 수직 투영
    vertical_projection = np.sum(binary_img, axis=0)
    
    if self.debug:
        plt.figure(figsize=(15, 4))
        plt.plot(vertical_projection)
        plt.title("Full Image Vertical Projection")
        plt.xlabel("X coordinate")
        plt.ylabel("Pixel sum")
        plt.show()
    
    # 매우 낮은 임계값 (전체 높이의 1%만 있어도 후보)
    min_threshold = height * 255 * 0.01  # 전체 높이의 1%
    
    candidates = []
    for x in range(width):
        if vertical_projection[x] > min_threshold:
            candidates.append(x)
    
    if self.debug:
        print(f"Raw candidates from projection: {len(candidates)}")
    
    return candidates, vertical_projection
```

#### **2.2 로컬 피크 검출**
```python
def find_projection_peaks(self, vertical_projection, min_distance=5):
    """수직 투영에서 피크들을 찾기 (바라인 후보)"""
    
    # scipy.signal.find_peaks 사용하되 매우 관대한 조건
    peaks, properties = find_peaks(
        vertical_projection,
        height=np.max(vertical_projection) * 0.05,  # 최대값의 5%만 되어도 피크
        distance=min_distance,  # 최소 5픽셀 간격
        prominence=np.max(vertical_projection) * 0.02  # 매우 낮은 prominence
    )
    
    # 피크 정보 상세 분석
    peak_info = []
    for peak_x in peaks:
        info = {
            'x': peak_x,
            'height': vertical_projection[peak_x],
            'prominence': properties['prominences'][list(peaks).index(peak_x)] if 'prominences' in properties else 0,
            'width_estimate': self._estimate_peak_width(vertical_projection, peak_x)
        }
        peak_info.append(info)
    
    if self.debug:
        print(f"Found {len(peaks)} projection peaks")
        for i, info in enumerate(peak_info):
            print(f"  Peak {i+1}: x={info['x']}, height={info['height']:.1f}, width≈{info['width_estimate']}")
    
    return peak_info

def _estimate_peak_width(self, projection, peak_x, ratio=0.5):
    """피크의 대략적인 폭 추정"""
    peak_height = projection[peak_x]
    threshold = peak_height * ratio
    
    # 왼쪽으로 스캔
    left_x = peak_x
    for x in range(peak_x - 1, -1, -1):
        if projection[x] < threshold:
            break
        left_x = x
    
    # 오른쪽으로 스캔  
    right_x = peak_x
    for x in range(peak_x + 1, len(projection)):
        if projection[x] < threshold:
            break
        right_x = x
    
    return right_x - left_x + 1
```

#### **2.3 스태프별 로컬 투영**
```python
def detect_by_staff_local_projection(self, binary_img):
    """각 스태프 영역별로 로컬 수직 투영 수행"""
    
    if len(self.staff_lines) < 3:
        return []
    
    # 스태프 그룹핑 (5개씩 또는 전체를 하나로)
    staff_groups = self._simple_staff_grouping()
    
    all_candidates = []
    
    for group_idx, staff_group in enumerate(staff_groups):
        # 스태프 영역 정의 (여유있게)
        staff_top = min(staff_group) - 10
        staff_bottom = max(staff_group) + 10
        staff_top = max(0, staff_top)
        staff_bottom = min(binary_img.shape[0], staff_bottom)
        
        # 스태프 영역만 추출
        staff_roi = binary_img[staff_top:staff_bottom, :]
        
        # 로컬 수직 투영
        local_projection = np.sum(staff_roi, axis=0)
        
        # 로컬 피크 검출 (더욱 관대하게)
        local_peaks = find_peaks(
            local_projection,
            height=np.max(local_projection) * 0.03,  # 3%만 되어도 피크
            distance=3  # 3픽셀 간격
        )[0]
        
        # 전체 좌표계로 변환하여 저장
        for peak_x in local_peaks:
            candidate_info = {
                'x': peak_x,
                'staff_group': group_idx,
                'local_height': local_projection[peak_x],
                'staff_top': staff_top,
                'staff_bottom': staff_bottom,
                'coverage_ratio': local_projection[peak_x] / (staff_bottom - staff_top) / 255
            }
            all_candidates.append(candidate_info)
        
        if self.debug:
            print(f"Staff group {group_idx}: {len(local_peaks)} local candidates")
    
    return all_candidates

def _simple_staff_grouping(self):
    """단순한 스태프 그룹핑 (5개씩 또는 전체)"""
    
    if len(self.staff_lines) <= 5:
        return [self.staff_lines]
    
    groups = []
    for i in range(0, len(self.staff_lines), 5):
        group = self.staff_lines[i:i+5]
        if len(group) >= 3:  # 최소 3개 라인
            groups.append(group)
    
    return groups
```

### **Phase 3: 최소한의 후처리**

#### **3.1 단순 중복 제거**
```python
def merge_nearby_candidates_simple(self, candidates, merge_distance=8):
    """가까운 후보들을 단순하게 병합"""
    
    if not candidates:
        return []
    
    # X좌표로 정렬
    sorted_candidates = sorted(candidates, key=lambda c: c['x'])
    
    merged = []
    current_group = [sorted_candidates[0]]
    
    for candidate in sorted_candidates[1:]:
        # 마지막 그룹의 대표 X좌표와 비교
        last_x = np.mean([c['x'] for c in current_group])
        
        if candidate['x'] - last_x <= merge_distance:
            current_group.append(candidate)
        else:
            # 현재 그룹 완료
            merged_candidate = self._merge_candidate_group(current_group)
            merged.append(merged_candidate)
            current_group = [candidate]
    
    # 마지막 그룹 처리
    if current_group:
        merged_candidate = self._merge_candidate_group(current_group)
        merged.append(merged_candidate)
    
    return merged

def _merge_candidate_group(self, group):
    """후보 그룹을 하나의 대표 후보로 병합"""
    
    merged = {
        'x': int(np.mean([c['x'] for c in group])),
        'confidence': len(group),  # 그룹 크기가 신뢰도
        'max_height': max([c.get('local_height', 0) for c in group]),
        'staff_groups': list(set([c.get('staff_group', 0) for c in group])),
        'source_count': len(group)
    }
    
    return merged
```

#### **3.2 극단적으로 관대한 필터링**
```python
def filter_candidates_liberal(self, candidates):
    """극도로 관대한 필터링 (명백히 잘못된 것만 제거)"""
    
    filtered = []
    
    for candidate in candidates:
        # 1. 이미지 경계 체크
        if candidate['x'] < 10 or candidate['x'] > self.binary_img.shape[1] - 10:
            continue  # 가장자리 너무 가까이는 제외
        
        # 2. 최소 신뢰도 체크 (매우 낮은 기준)
        if candidate['confidence'] < 1:  # 적어도 1개 소스에서 검출
            continue
        
        # 3. 그 외에는 모두 통과
        filtered.append(candidate)
    
    if self.debug:
        print(f"Liberal filtering: {len(candidates)} → {len(filtered)} candidates")
    
    return filtered
```

#### **3.3 스태프 교차 체크 (선택사항)**
```python
def optional_staff_intersection_check(self, candidates):
    """선택적 스태프 교차 확인 (너무 관대해서 많이 검출될 때만)"""
    
    if len(candidates) < 20:  # 후보가 적으면 스킵
        return candidates
    
    # 후보가 너무 많을 때만 스태프 교차로 필터링
    staff_verified = []
    
    for candidate in candidates:
        x = candidate['x']
        intersections = 0
        
        # 스태프와의 교차 체크 (매우 관대하게)
        for staff_y in self.staff_lines:
            if self._has_any_intersection_at_position(x, staff_y):
                intersections += 1
        
        # 최소 2개 스태프와 교차하면 유지
        if intersections >= 2:
            candidate['staff_intersections'] = intersections
            staff_verified.append(candidate)
    
    if self.debug:
        print(f"Optional staff check: {len(candidates)} → {len(staff_verified)} candidates")
    
    # 필터링 결과가 너무 적으면 원본 반환
    return staff_verified if len(staff_verified) >= 3 else candidates

def _has_any_intersection_at_position(self, x, staff_y):
    """특정 위치에 아무 교차점이라도 있는지 확인"""
    
    # 매우 넓은 범위에서 확인
    y_start = max(0, staff_y - 5)
    y_end = min(self.binary_img.shape[0], staff_y + 6)
    x_start = max(0, x - 3)
    x_end = min(self.binary_img.shape[1], x + 4)
    
    roi = self.binary_img[y_start:y_end, x_start:x_end]
    
    # 아무 검은 픽셀이라도 있으면 교차로 인정
    return np.any(roi > 0)
```

### **Phase 4: 통합 시스템**

#### **4.1 메인 단순 검출 함수**
```python
def detect_barlines_simple(self, binary_img):
    """단순 수직 투영 방식 바라인 검출"""
    
    if self.debug:
        print("=== 단순 수직 투영 방식 바라인 검출 시작 ===")
    
    # Phase 1: 최소한의 전처리
    processed_img = self.try_multiple_thresholds(binary_img)
    
    # Phase 2: 다양한 방법으로 후보 수집
    # 방법 1: 전체 이미지 수직 투영
    global_candidates, projection = self.detect_all_vertical_content(processed_img)
    global_peaks = self.find_projection_peaks(projection)
    
    # 방법 2: 스태프별 로컬 투영
    local_candidates = self.detect_by_staff_local_projection(processed_img)
    
    # 모든 후보를 통합
    all_candidates = []
    
    # Global peaks를 후보 형식으로 변환
    for peak in global_peaks:
        all_candidates.append({
            'x': peak['x'],
            'source': 'global',
            'local_height': peak['height'],
            'confidence': 1
        })
    
    # Local candidates 추가
    for candidate in local_candidates:
        all_candidates.append({
            'x': candidate['x'],
            'source': 'local',
            'local_height': candidate['local_height'],
            'confidence': 1
        })
    
    if self.debug:
        print(f"Total raw candidates: {len(all_candidates)}")
    
    # Phase 3: 최소한의 후처리
    merged_candidates = self.merge_nearby_candidates_simple(all_candidates)
    filtered_candidates = self.filter_candidates_liberal(merged_candidates)
    
    # 선택적 스태프 교차 체크
    final_candidates = self.optional_staff_intersection_check(filtered_candidates)
    
    # X좌표만 추출
    barline_positions = sorted([c['x'] for c in final_candidates])
    
    if self.debug:
        print(f"=== 최종 결과: {len(barline_positions)}개 바라인 후보 ===")
        for i, x in enumerate(barline_positions):
            print(f"  후보 {i+1}: x = {x}")
    
    return barline_positions
```

#### **4.2 백업 극단 모드**
```python
def detect_barlines_extreme_liberal(self, binary_img):
    """극단적으로 관대한 백업 검출 (모든 것이 실패했을 때)"""
    
    if self.debug:
        print("=== 극단 관대 모드 활성화 ===")
    
    height, width = binary_img.shape
    candidates = []
    
    # 모든 X좌표에 대해 수직 픽셀 수 계산
    for x in range(0, width, 2):  # 2픽셀마다 체크 (성능상)
        column = binary_img[:, x]
        black_pixels = np.sum(column > 0)
        
        # 전체 높이의 0.5%만 있어도 후보
        if black_pixels > height * 0.005:
            candidates.append(x)
    
    # 간단한 클러스터링으로 인접한 것들 병합
    if not candidates:
        return []
    
    clustered = []
    current_group = [candidates[0]]
    
    for x in candidates[1:]:
        if x - current_group[-1] <= 10:  # 10픽셀 이내
            current_group.append(x)
        else:
            # 그룹 완료
            center = int(np.mean(current_group))
            clustered.append(center)
            current_group = [x]
    
    # 마지막 그룹
    if current_group:
        center = int(np.mean(current_group))
        clustered.append(center)
    
    if self.debug:
        print(f"극단 관대 모드 결과: {len(clustered)}개 후보")
    
    return clustered
```

---

## 📊 성능 예상

### **예상 결과**
- **검출율**: 95-99% (거의 모든 바라인 검출)
- **정확도**: 30-60% (많은 False Positive 포함)
- **속도**: 매우 빠름 (단순한 연산만 사용)
- **메모리**: 적음

### **사용 시나리오**
1. **디버깅 목적**: 바라인이 어디에 있는지 일단 확인
2. **다른 방법 실패시**: HoughLinesP, 하이브리드 방법 모두 실패했을 때
3. **사용자 가이드**: 사용자가 수동으로 선택할 수 있도록 모든 후보 제시

### **후처리 권장사항**
이 방법으로 얻은 결과는 다음과 같은 추가 처리가 필요:
1. **사용자 검증**: 사용자가 올바른 바라인 선택
2. **규칙 기반 필터링**: 마디 길이 분포 등으로 후필터링
3. **다른 방법과 교차 검증**: 여러 방법 결과의 교집합 사용

---

## 🔧 구현 순서

### **Day 1: 기본 투영 검출**
- [ ] `try_multiple_thresholds()` 구현
- [ ] `detect_all_vertical_content()` 구현
- [ ] 기본 투영 결과 확인

### **Day 2: 피크 검출 및 로컬 분석**  
- [ ] `find_projection_peaks()` 구현
- [ ] `detect_by_staff_local_projection()` 구현
- [ ] 다양한 방법 결과 비교

### **Day 3: 후처리 및 통합**
- [ ] `merge_nearby_candidates_simple()` 구현
- [ ] `detect_barlines_simple()` 통합
- [ ] La Gazza ladra Overture 테스트

### **Day 4: 백업 시스템**
- [ ] `detect_barlines_extreme_liberal()` 구현
- [ ] 전체 시스템 테스트
- [ ] 성능 비교 분석

---

이 방법은 **"확실히 놓치지는 않겠다"**는 보장을 제공하며, 다른 정교한 방법들이 실패했을 때의 **안전망** 역할을 합니다.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"id": "create_hough_plan", "content": "Create detailed implementation plan for HoughLinesP-based barline detection", "status": "completed", "priority": "high"}, {"id": "create_hybrid_plan", "content": "Create implementation plan for hybrid approach (Morphological + Hough)", "status": "completed", "priority": "medium"}, {"id": "create_simple_plan", "content": "Create implementation plan for simple vertical projection approach", "status": "completed", "priority": "low"}]