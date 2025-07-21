# 하이브리드 접근법: 형태학적 전처리 + HoughLinesP

**작성일**: 2025년 7월 21일  
**문서 목적**: 형태학적 연산으로 후보를 추려내고 HoughLinesP로 정밀 검출하는 하이브리드 시스템 설계  
**우선순위**: #2 (속도와 정확도의 균형)

---

## 🎯 전략 개요

HoughLinesP만으로는 과검출 위험이 있고, 형태학적 연산만으로는 정밀도가 부족합니다. **두 방법의 장점을 결합**하여:

1. **형태학적 연산**으로 빠르게 수직 후보 영역 추출
2. **HoughLinesP**로 후보 영역 내에서 정밀한 선분 검출
3. **기하학적 분석**으로 최종 바라인 결정

### **핵심 아이디어**
- 형태학적 연산이 "관심 영역(ROI)"을 제공
- HoughLinesP가 ROI 내에서만 동작하여 연산 효율성 확보
- 단계별 필터링으로 False Positive 최소화

---

## 🔧 상세 구현 계획

### **Phase 1: 형태학적 전처리로 후보 영역 추출**

#### **1.1 적응적 수직 커널 생성**
```python
def create_adaptive_vertical_kernel(self):
    """스태프 간격에 기반한 적응적 수직 커널 생성"""
    
    if len(self.staff_lines) >= 2:
        # 스태프 간격 분석
        spacings = [self.staff_lines[i+1] - self.staff_lines[i] 
                   for i in range(len(self.staff_lines)-1) 
                   if self.staff_lines[i+1] - self.staff_lines[i] < 30]
        
        avg_spacing = np.median(spacings) if spacings else 12
        
        # 커널 높이: 스태프 간격의 80% (오선간 공백을 피하되 교차점은 포함)
        kernel_height = max(8, int(avg_spacing * 0.8))
    else:
        kernel_height = 15  # 기본값
    
    # 다양한 두께의 바라인을 고려한 커널 세트
    kernels = {
        'thin': cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height)),
        'medium': cv2.getStructuringElement(cv2.MORPH_RECT, (2, kernel_height)),
        'thick': cv2.getStructuringElement(cv2.MORPH_RECT, (3, kernel_height))
    }
    
    return kernels
```

#### **1.2 다중 커널 형태학적 검출**
```python
def extract_vertical_candidates_morphology(self, binary_img):
    """다중 커널을 사용한 수직 후보 영역 추출"""
    
    kernels = self.create_adaptive_vertical_kernel()
    candidate_masks = []
    
    # 각 커널별로 수직 요소 추출
    for kernel_name, kernel in kernels.items():
        # Opening 연산으로 수직선 추출
        vertical_mask = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
        
        # 약간의 확장으로 후보 영역 여유 확보
        dilated_mask = cv2.dilate(vertical_mask, 
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), 
                                 iterations=1)
        
        candidate_masks.append(dilated_mask)
        
        if self.debug:
            cv2.imshow(f"Morphology - {kernel_name}", 
                      cv2.resize(vertical_mask, None, fx=0.3, fy=0.3))
    
    # 모든 마스크 통합 (OR 연산)
    combined_mask = np.zeros_like(binary_img)
    for mask in candidate_masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    if self.debug:
        cv2.imshow("Combined Morphology Mask", 
                  cv2.resize(combined_mask, None, fx=0.3, fy=0.3))
        cv2.waitKey(0)
    
    return combined_mask
```

#### **1.3 관심 영역(ROI) 추출**
```python
def extract_vertical_rois(self, candidate_mask):
    """후보 마스크에서 관심 영역들 추출"""
    
    # 수직 projection으로 후보 x좌표들 찾기
    vertical_projection = np.sum(candidate_mask, axis=0)
    
    # 임계값 이상인 영역들 찾기
    threshold = np.max(vertical_projection) * 0.1  # 관대한 임계값
    candidate_columns = np.where(vertical_projection > threshold)[0]
    
    if len(candidate_columns) == 0:
        return []
    
    # 연속된 컬럼들을 ROI로 그룹핑
    rois = []
    roi_start = candidate_columns[0]
    
    for i in range(1, len(candidate_columns)):
        if candidate_columns[i] - candidate_columns[i-1] > 5:  # 5픽셀 이상 갭
            # 현재 ROI 완료
            roi_end = candidate_columns[i-1]
            roi_center = (roi_start + roi_end) // 2
            roi_width = max(10, roi_end - roi_start + 6)  # 최소 10픽셀 너비
            
            rois.append({
                'center_x': roi_center,
                'x_start': max(0, roi_center - roi_width // 2),
                'x_end': min(candidate_mask.shape[1], roi_center + roi_width // 2),
                'projection_strength': np.sum(vertical_projection[roi_start:roi_end+1])
            })
            
            # 새로운 ROI 시작
            roi_start = candidate_columns[i]
    
    # 마지막 ROI 처리
    roi_end = candidate_columns[-1]
    roi_center = (roi_start + roi_end) // 2
    roi_width = max(10, roi_end - roi_start + 6)
    rois.append({
        'center_x': roi_center,
        'x_start': max(0, roi_center - roi_width // 2),
        'x_end': min(candidate_mask.shape[1], roi_center + roi_width // 2),
        'projection_strength': np.sum(vertical_projection[roi_start:roi_end+1])
    })
    
    if self.debug:
        print(f"Extracted {len(rois)} ROIs from morphological analysis")
    
    return rois
```

### **Phase 2: ROI별 HoughLinesP 정밀 검출**

#### **2.1 ROI별 HoughLinesP 적용**
```python
def detect_lines_in_rois(self, binary_img, rois):
    """각 ROI 내에서 HoughLinesP로 선분 검출"""
    
    all_detected_lines = []
    
    for roi_idx, roi in enumerate(rois):
        # ROI 영역 추출
        roi_img = binary_img[:, roi['x_start']:roi['x_end']]
        
        if roi_img.shape[1] < 5:  # 너무 좁은 ROI는 스킵
            continue
        
        # ROI 크기에 맞는 파라미터 조정
        roi_width = roi['x_end'] - roi['x_start']
        
        # 파라미터 자동 조정
        params = self._get_roi_hough_params(roi_img, roi)
        
        # HoughLinesP 적용
        lines = cv2.HoughLinesP(
            roi_img,
            rho=1,
            theta=np.pi/180,
            threshold=params['threshold'],
            minLineLength=params['minLineLength'], 
            maxLineGap=params['maxLineGap']
        )
        
        if lines is not None:
            # ROI 좌표를 전체 이미지 좌표로 변환
            global_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                global_x1 = x1 + roi['x_start']
                global_x2 = x2 + roi['x_start']
                
                global_lines.append({
                    'line': [global_x1, y1, global_x2, y2],
                    'roi_id': roi_idx,
                    'roi_center': roi['center_x'],
                    'length': np.sqrt((x2-x1)**2 + (y2-y1)**2),
                    'angle': np.arctan2(y2-y1, x2-x1) * 180/np.pi if x2 != x1 else 90
                })
            
            all_detected_lines.extend(global_lines)
            
            if self.debug:
                print(f"ROI {roi_idx} (center={roi['center_x']}): "
                      f"detected {len(lines)} lines")
    
    return all_detected_lines
```

#### **2.2 ROI별 적응적 파라미터**
```python
def _get_roi_hough_params(self, roi_img, roi_info):
    """ROI 특성에 따른 HoughLinesP 파라미터 계산"""
    
    height, width = roi_img.shape
    pixel_density = np.sum(roi_img > 0) / (height * width)
    
    # 기본 파라미터
    base_params = {
        'threshold': 8,
        'minLineLength': 5,
        'maxLineGap': 3
    }
    
    # ROI 크기 기반 조정
    if width < 10:  # 좁은 ROI
        base_params['threshold'] = max(3, base_params['threshold'] // 2)
        base_params['minLineLength'] = max(3, base_params['minLineLength'] - 2)
    
    # 픽셀 밀도 기반 조정
    if pixel_density > 0.3:  # 고밀도 영역
        base_params['threshold'] = min(15, base_params['threshold'] + 5)
    elif pixel_density < 0.1:  # 저밀도 영역
        base_params['threshold'] = max(3, base_params['threshold'] - 3)
    
    # ROI projection 강도 기반 조정
    strength_ratio = roi_info['projection_strength'] / (height * width)
    if strength_ratio > 0.5:
        base_params['minLineLength'] = max(3, int(base_params['minLineLength'] * 0.8))
    
    return base_params
```

### **Phase 3: 기하학적 분석 및 최종 선별**

#### **3.1 선분 클러스터링 및 분석**
```python
def cluster_and_analyze_lines(self, detected_lines):
    """검출된 선분들을 클러스터링하고 분석"""
    
    if not detected_lines:
        return []
    
    # ROI별로 선분들 그룹핑 (이미 ROI 정보가 있음)
    roi_groups = {}
    for line_info in detected_lines:
        roi_id = line_info['roi_id']
        if roi_id not in roi_groups:
            roi_groups[roi_id] = []
        roi_groups[roi_id].append(line_info)
    
    analyzed_barlines = []
    
    for roi_id, lines in roi_groups.items():
        # 이 ROI 내 선분들 분석
        analysis = self._analyze_roi_lines(lines)
        
        if analysis['is_barline_candidate']:
            analyzed_barlines.append(analysis)
    
    return analyzed_barlines

def _analyze_roi_lines(self, lines):
    """ROI 내 선분들의 바라인 특성 분석"""
    
    # 기본 통계
    center_x = np.mean([l['roi_center'] for l in lines])
    angles = [abs(l['angle']) for l in lines]
    lengths = [l['length'] for l in lines]
    y_positions = []
    
    for line in lines:
        x1, y1, x2, y2 = line['line']
        y_positions.extend([y1, y2])
    
    analysis = {
        'center_x': center_x,
        'line_count': len(lines),
        'total_length': sum(lengths),
        'avg_length': np.mean(lengths),
        'angle_consistency': np.std(angles),  # 각도 일관성
        'y_coverage': max(y_positions) - min(y_positions) if y_positions else 0,
        'verticality_score': np.mean([90 - abs(90 - abs(a)) for a in angles])  # 수직성 점수
    }
    
    # 바라인 가능성 점수 계산
    score = 0
    
    # 1. 수직성 (각도 일관성)
    if analysis['angle_consistency'] < 10:
        score += 25
    elif analysis['angle_consistency'] < 20:
        score += 15
    
    # 2. 수직도 (90도에 가까움)
    if analysis['verticality_score'] > 80:
        score += 30
    elif analysis['verticality_score'] > 70:
        score += 20
    
    # 3. Y축 커버리지 (스태프 영역 커버)
    expected_staff_height = self._estimate_staff_height()
    coverage_ratio = analysis['y_coverage'] / expected_staff_height if expected_staff_height > 0 else 0
    
    if coverage_ratio > 0.6:
        score += 25
    elif coverage_ratio > 0.4:
        score += 15
    
    # 4. 선분 개수 (적절한 개수)
    if 2 <= analysis['line_count'] <= 6:
        score += 20
    elif analysis['line_count'] >= 1:
        score += 10
    
    analysis['barline_score'] = score
    analysis['is_barline_candidate'] = score >= 40  # 40점 이상이면 후보
    
    return analysis
```

#### **3.2 스태프 교차 검증**
```python
def verify_staff_intersections_hybrid(self, barline_candidates):
    """스태프와의 교차점 검증 (하이브리드 방식)"""
    
    verified_barlines = []
    
    for candidate in barline_candidates:
        center_x = int(candidate['center_x'])
        
        # 스태프 교차점 확인
        intersections = 0
        intersection_details = []
        
        for staff_y in self.staff_lines:
            # 교차점 검사를 위한 넓은 범위 (±5픽셀)
            intersection_strength = self._check_intersection_strength(center_x, staff_y)
            
            if intersection_strength > 0.3:  # 30% 이상 교차
                intersections += 1
                intersection_details.append({
                    'staff_y': staff_y,
                    'strength': intersection_strength
                })
        
        # 검증 기준
        staff_groups = self._group_staff_lines_for_verification()
        valid = False
        
        for staff_group in staff_groups:
            group_intersections = sum(1 for detail in intersection_details 
                                    if detail['staff_y'] in staff_group)
            
            # 스태프 그룹의 60% 이상과 교차하면 유효
            if group_intersections >= len(staff_group) * 0.6:
                valid = True
                break
        
        if valid:
            candidate['staff_intersections'] = intersections
            candidate['intersection_details'] = intersection_details
            candidate['verification_passed'] = True
            verified_barlines.append(candidate)
        
        if self.debug:
            print(f"Barline at x={center_x}: {intersections} intersections, "
                  f"valid={valid}")
    
    return verified_barlines

def _check_intersection_strength(self, x, staff_y):
    """특정 위치에서의 교차 강도 계산"""
    
    # 스태프 라인 주변 ±3픽셀, x좌표 주변 ±2픽셀 영역
    y_start = max(0, staff_y - 3)
    y_end = min(self.binary_img.shape[0], staff_y + 4)
    x_start = max(0, x - 2)
    x_end = min(self.binary_img.shape[1], x + 3)
    
    roi = self.binary_img[y_start:y_end, x_start:x_end]
    
    # 교차 강도 = (검은 픽셀 수) / (전체 픽셀 수)
    total_pixels = roi.shape[0] * roi.shape[1]
    black_pixels = np.sum(roi > 0)
    
    return black_pixels / total_pixels if total_pixels > 0 else 0
```

### **Phase 4: 통합 및 최종 시스템**

#### **4.1 메인 하이브리드 검출 함수**
```python
def detect_barlines_hybrid(self, binary_img):
    """하이브리드 방식 바라인 검출 메인 함수"""
    
    if self.debug:
        print("=== 하이브리드 바라인 검출 시작 ===")
    
    # Phase 1: 형태학적 전처리로 후보 추출
    candidate_mask = self.extract_vertical_candidates_morphology(binary_img)
    rois = self.extract_vertical_rois(candidate_mask)
    
    if not rois:
        if self.debug:
            print("형태학적 분석에서 후보 영역을 찾지 못함")
        return []
    
    # Phase 2: ROI별 HoughLinesP 검출
    detected_lines = self.detect_lines_in_rois(binary_img, rois)
    
    if not detected_lines:
        if self.debug:
            print("HoughLinesP에서 선분을 찾지 못함")
        return []
    
    # Phase 3: 기하학적 분석
    barline_candidates = self.cluster_and_analyze_lines(detected_lines)
    
    if not barline_candidates:
        if self.debug:
            print("기하학적 분석에서 바라인 후보를 찾지 못함")
        return []
    
    # Phase 4: 스태프 교차 검증
    verified_barlines = self.verify_staff_intersections_hybrid(barline_candidates)
    
    # 최종 결과 정렬
    final_barlines = sorted([b['center_x'] for b in verified_barlines])
    
    if self.debug:
        print(f"=== 최종 결과: {len(final_barlines)}개 바라인 검출 ===")
        for i, x in enumerate(final_barlines):
            print(f"  바라인 {i+1}: x = {x}")
    
    return final_barlines
```

#### **4.2 성능 최적화**
```python
def optimize_hybrid_performance(self):
    """하이브리드 방식 성능 최적화 설정"""
    
    # 이미지 크기에 따른 처리 전략
    height, width = self.binary_img.shape
    
    if width > 2000:  # 고해상도 이미지
        self.hybrid_config = {
            'morphology_iterations': 1,      # 형태학적 연산 최소화
            'roi_max_count': 50,            # ROI 개수 제한
            'hough_threshold_boost': 1.5,   # Hough 임계값 상향
            'parallel_processing': True     # 병렬 처리 활성화
        }
    else:  # 일반 해상도
        self.hybrid_config = {
            'morphology_iterations': 2,
            'roi_max_count': 100,
            'hough_threshold_boost': 1.0,
            'parallel_processing': False
        }
    
    if self.debug:
        print(f"하이브리드 최적화 설정: {self.hybrid_config}")
```

---

## 📊 성능 예상 및 비교

### **방법별 성능 비교**

| 특성 | 세그먼트 방식 | HoughLinesP 전용 | 하이브리드 방식 |
|------|--------------|-----------------|----------------|
| **검출율** | 0% | 85-95% | 80-90% |
| **정확도** | - | 85-90% | 90-95% |
| **속도** | 빠름 | 느림 | 보통 |
| **메모리** | 적음 | 많음 | 보통 |
| **강건성** | 낮음 | 보통 | 높음 |
| **구현 복잡도** | 보통 | 낮음 | 높음 |

### **하이브리드 방식의 장점**
1. **높은 정확도**: 2단계 필터링으로 False Positive 최소화
2. **효율적 연산**: ROI 기반 처리로 전체 이미지 HoughLinesP보다 빠름
3. **강건성**: 다양한 이미지 품질에 안정적 대응
4. **확장성**: 각 단계별 독립적 최적화 가능

---

## 🔧 구현 순서

### **Sprint 1 (3-4일): 형태학적 전처리**
- [ ] `create_adaptive_vertical_kernel()` 구현
- [ ] `extract_vertical_candidates_morphology()` 구현  
- [ ] `extract_vertical_rois()` 구현
- [ ] 형태학적 단계 단독 테스트

### **Sprint 2 (3-4일): ROI별 HoughLinesP**
- [ ] `detect_lines_in_rois()` 구현
- [ ] `_get_roi_hough_params()` 구현
- [ ] ROI별 선분 검출 테스트
- [ ] 성능 프로파일링

### **Sprint 3 (4-5일): 분석 및 검증**
- [ ] `cluster_and_analyze_lines()` 구현
- [ ] `verify_staff_intersections_hybrid()` 구현
- [ ] 통합 테스트 및 디버깅

### **Sprint 4 (2-3일): 최적화 및 배포**
- [ ] `optimize_hybrid_performance()` 구현
- [ ] 다양한 악보 테스트
- [ ] 메인 시스템 통합

---

이 하이브리드 접근법은 **정확도와 효율성의 최적 균형점**을 제공하며, 실제 악보에서 높은 성능을 보일 것으로 예상됩니다.