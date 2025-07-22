# Bracket 검출 기술 계획: 하이브리드 접근법

**작성일**: 2025년 7월 22일
**문서 목적**: 악보 좌측의 System Group을 묶는 대괄호(Bracket)를 검출하기 위한 구체적이고 기술적인 실행 계획 수립

---

## 🎯 목표

- 악보 이미지의 가장 왼쪽에 위치하며 여러 보표(staff systems)를 하나로 묶는 **Bracket을 정확하게 검출**한다.
- 검출된 Bracket의 **정확한 위치(x, y_start, y_end)와 이것이 포괄하는 보표들의 인덱스 정보를 추출**한다.
- 이 정보를 기존의 `system_groups` 정보와 상호 검증하여 보표 그룹 인식의 정확도를 높인다.

---

## 🚀 핵심 전략: Hough 변환과 템플릿 매칭의 하이브리드 접근법

Bracket의 두 가지 주요 시각적 특징인 **① 긴 수직선**과 **② 뾰족한 모서리**를 각각 다른 기술로 검출하고 조합하여 신뢰도를 극대화한다.

1.  **1단계 (거시적 접근)**: **Hough Line Transform**으로 이미지 전체에서 긴 수직선을 찾아 Bracket의 몸통 후보를 식별한다.
2.  **2단계 (미시적 접근)**: **Template Matching**으로 1단계에서 찾은 후보의 끝점에서 Bracket 특유의 뾰족한 모서리를 찾아 검증한다.

---

## 🔧 상세 기술 구현 계획

### **Phase 1: 후보 영역 식별 및 긴 수직선 검출**

#### 1.1. 관심 영역(ROI) 최적화
- Bracket은 항상 악보의 왼쪽에 위치하므로, 전체 이미지를 탐색할 필요가 없다.
- **ROI 설정**:
  - **X축**: 이미지 너비의 **0% ~ 15%** 영역. (대부분의 경우 10% 내에 위치)
  - **Y축**: 검출된 모든 보표(`staff_lines`)의 최상단(`min(y)`)부터 최하단(`max(y)`)까지.
  - 이 ROI를 설정함으로써, 악보의 주요 내용과 분리하여 탐색 속도를 높이고 오검출을 줄인다.

#### 1.2. Hough Line Transform을 이용한 수직선 검출
- **목표**: 여러 보표를 관통하는 긴 수직선만 필터링.
- **함수**: `cv2.HoughLinesP()`
- **핵심 파라미터 설정**:
  - `rho`: `1` (1픽셀 정밀도)
  - `theta`: `np.pi / 180` (1도 정밀도)
  - `threshold`: `50` ~ `100` 사이의 값. ROI 내의 픽셀 수에 따라 동적으로 조절 가능.
  - **`minLineLength` (매우 중요)**: Bracket 몸통의 최소 길이를 정의. 이 값을 통해 짧은 마디선이나 다른 수직 노이즈를 효과적으로 제거.
    - **계산식**: `min_staff_system_height * 1.5` (최소 1.5개의 보표 그룹을 관통하는 길이). 또는 `(max(staff_y) - min(staff_y)) * 0.5` (전체 보표 높이의 50% 이상).
  - **`maxLineGap`**: `avg_staff_spacing * 0.5` (평균 보표 간격의 50%) 정도로 설정하여, 오선으로 인해 생기는 단절을 허용.

- **코드 예시**:
  ```python
  def find_vertical_bracket_candidates(roi_image, staff_systems):
      # minLineLength 동적 계산
      if not staff_systems:
          return []
      min_height = min(s['height'] for s in staff_systems)
      min_line_length = min_height * 1.5

      # maxLineGap 동적 계산
      avg_spacing = np.mean([s['avg_spacing'] for s in staff_systems])
      max_line_gap = avg_spacing * 0.5

      lines = cv2.HoughLinesP(
          roi_image,
          rho=1,
          theta=np.pi / 180,
          threshold=80,
          minLineLength=min_line_length,
          maxLineGap=max_line_gap
      )

      # 수직선 필터링 (88~92도)
      vertical_candidates = []
      if lines is not None:
          for line in lines:
              x1, y1, x2, y2 = line[0]
              angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
              if 88 < angle < 92:
                  vertical_candidates.append(line[0])
      
      return vertical_candidates
  ```

### **Phase 2: 템플릿 매칭을 통한 모서리 검증**

#### 2.1. 템플릿 이미지 준비
- **소스**: 실제 악보 이미지에서 고해상도의 Bracket 모서리 부분을 잘라내어 템플릿으로 사용.
- **종류**: `template_bracket_top.png`, `template_bracket_bottom.png` 두 종류를 준비.
- **전처리**: 템플릿 이미지는 배경이 투명하거나 흰색이어야 하며, 이진화(binary)된 상태가 매칭에 유리.
- **다중 스케일 대응**: `for scale in np.linspace(0.8, 1.2, 10):` 와 같이 템플릿의 크기를 약간씩 변경하며 매칭을 시도하여, 다양한 크기의 Bracket에 대응한다.

#### 2.2. 템플릿 매칭 실행
- **목표**: Phase 1에서 찾은 수직선 후보의 양 끝점에서 뾰족한 모서리 발견.
- **함수**: `cv2.matchTemplate()`
- **프로세스**:
  1.  각 수직선 후보(`x1, y1, x2, y2`)에 대해, 상단 끝점(`y1`)과 하단 끝점(`y2`) 주변을 탐색 ROI로 설정한다. (예: `30x30` 픽셀 크기)
  2.  상단 ROI에서는 `template_bracket_top.png`와 매칭, 하단 ROI에서는 `template_bracket_bottom.png`와 매칭을 시도한다.
  3.  매칭 방법(method)은 `cv2.TM_CCOEFF_NORMED` 를 사용한다. 이 방법은 밝기 변화에 강건하고 정규화된 일치도 점수(0~1)를 반환한다.
  4.  `cv2.minMaxLoc()` 함수로 매칭 결과에서 최대 일치도 점수와 위치를 찾는다.

- **코드 예시**:
  ```python
  def verify_bracket_corners(image, vertical_line, top_template, bottom_template):
      x1, y1, x2, y2 = vertical_line
      
      # 상단 모서리 검증
      top_roi = image[y1 - 15 : y1 + 15, x1 - 15 : x1 + 15]
      res_top = cv2.matchTemplate(top_roi, top_template, cv2.TM_CCOEFF_NORMED)
      _, max_val_top, _, _ = cv2.minMaxLoc(res_top)
      
      # 하단 모서리 검증
      bottom_roi = image[y2 - 15 : y2 + 15, x1 - 15 : x1 + 15]
      res_bottom = cv2.matchTemplate(bottom_roi, bottom_template, cv2.TM_CCOEFF_NORMED)
      _, max_val_bottom, _, _ = cv2.minMaxLoc(res_bottom)
      
      # 임계값 (Threshold) 기반 최종 판정
      corner_threshold = 0.7  # 70% 이상 일치해야 함
      top_verified = max_val_top >= corner_threshold
      bottom_verified = max_val_bottom >= corner_threshold
      
      return top_verified and bottom_verified
  ```

### **Phase 3: 최종 결정 및 정보 종합**

- **최종 판정 로직**:
  - `find_vertical_bracket_candidates`를 통해 얻은 각 `line`에 대해 `verify_bracket_corners` 함수를 실행한다.
  - `verify_bracket_corners`가 `True`를 반환하는 `line`만이 최종적으로 검출된 Bracket이다.
- **정보 추출**:
  - **위치**: 검증된 `line`의 `x`좌표 (평균값), `y_start`, `y_end`를 기록한다.
  - **포괄 보표**: Bracket의 `y_start`와 `y_end` 범위 내에 완전히 포함되는 모든 `staff_system`의 인덱스를 리스트로 저장한다.
- **데이터 구조화**: 검출된 각 Bracket에 대해 다음 정보를 포함하는 객체를 생성한다.
  ```json
  {
    "type": "bracket",
    "bounding_box": { "x": 150, "y_start": 350, "y_end": 980 },
    "covered_staff_system_indices": [0, 1, 2, 3]
  }
  ```

---

## 🚨 위험 요소 및 대안 전략

1.  **위험**: 다양한 스타일과 크기의 Bracket에 템플릿이 모두 대응하지 못할 수 있다.
    - **대응**: 다중 스케일 템플릿 매칭을 필수로 적용하고, 필요시 여러 스타일의 템플릿을 준비한다.

2.  **위험**: 템플릿 매칭이 계산 비용이 높을 수 있다.
    - **대응**: Phase 1에서 ROI와 후보군을 최대한 압축하여, 템플릿 매칭이 최소한의 영역에서만 실행되도록 한다.

3.  **대안 전략**: **Contour 분석**
    - 템플릿 매칭이 잘 동작하지 않을 경우, Contour 분석으로 전환할 수 있다.
    - **프로세스**: ROI 내에서 `cv2.findContours()`로 윤곽선을 찾고, `cv2.approxPolyDP()`로 윤곽선을 다각형으로 근사화한다. Bracket은 보통 6~8개의 꼭짓점을 가진 특정 모양의 다각형으로 표현되므로, 이 꼭짓점의 개수와 상대적 위치, 각도를 분석하여 판별한다.
    - **장점**: 템플릿보다 다양한 형태에 유연하게 대응할 수 있다.
    - **단점**: 로직이 더 복잡하고, 노이즈에 민감할 수 있다.

---

## 📊 성공 지표

- `La Gazza ladra Overture` 악보의 모든 페이지에서 Bracket을 100% 검출한다.
- 검출된 Bracket이 포괄하는 보표 정보가 `detect_system_groups()`의 결과와 99% 이상 일치함을 확인한다.
- Bracket이 아닌 다른 객체(마디선, 텍스트 등)를 Bracket으로 오검출하는 비율(False Positive)이 1% 미만이어야 한다.
