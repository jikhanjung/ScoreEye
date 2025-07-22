# 마디 단위 이미지 추출 시스템 구현 완료 보고서

**작성일**: 2025년 7월 22일  
**문서 목적**: Phase 1.1 마디 단위 이미지 및 메타데이터 생성 시스템의 상세 구현 과정 및 결과 정리  
**관련 문서**: `devlog/20250722_03_comprehensive_omr_plan.md` Phase 1.1 구현

---

## 🎯 구현 목표 달성 현황

**✅ 완료된 목표**:
1. **마디 단위 이미지 자동 추출**: PDF → 페이지 → 시스템 그룹 → 개별 마디 이미지
2. **정제된 메타데이터 생성**: 라벨링 및 후처리에 필수적인 위치 정보 (BBox, 오선 좌표) JSON 파일
3. **CLI/GUI 통합 솔루션**: 명령행 도구와 시각적 GUI 인터페이스 모두 제공
4. **Consensus 검증 적용**: 후보 barlines가 아닌 검증된 barlines만 사용하여 정확도 확보

---

## 🚀 핵심 구현 결과

### **1. CLI 도구: `extract_measures.py`**

**기능**: 배치 처리용 명령행 도구
```bash
# 기본 사용법
python extract_measures.py input.pdf --output output/measures --dpi 300

# 특정 페이지만 처리
python extract_measures.py input.pdf -p 1 --debug

# 페이지 범위 지정
python extract_measures.py input.pdf -p 1-3
```

**핵심 구현 특징**:
- **PyMuPDF 기반**: poppler 의존성 제거로 설치 간소화
- **Consensus 검증**: `detect_barlines_per_system()` 메서드로 80% 이상 합의된 barlines만 사용
- **시스템 그룹 인식**: 4중주/앙상블 점수의 시스템 클러스터링 자동 처리
- **상대 좌표 매핑**: 마디 내 오선 위치를 상대 좌표로 변환하여 메타데이터 저장

### **2. GUI 통합 기능: `scoreeye_gui.py` 확장**

**새로 추가된 UI 컴포넌트**:
- **"Show Measure Boxes" 체크박스**: 실시간 마디 경계 미리보기
- **"Extract Measures" 버튼**: GUI에서 직접 마디 추출 실행

**GUI 워크플로우**:
1. PDF 로드 → 2. "Run Detection" → 3. "Show Measure Boxes" 체크 → 4. 미리보기 확인 → 5. "Extract Measures" 실행

**시각적 미리보기**: 
- 빨간색 세로선: Consensus 검증된 barlines
- 초록색 사각형: 추출될 마디 bounding boxes
- 시스템별 색상 구분: 각 오선 시스템을 다른 색으로 표시

---

## 🔧 상세 구현 과정 및 해결한 문제들

### **Phase 1: 초기 CLI 스크립트 개발**

#### **문제 1: poppler 의존성 이슈**
- **증상**: `pdf2image` 모듈 사용 시 "poppler가 설치되지 않음" 오류
- **해결책**: PyMuPDF로 완전 전환
```python
# Before (pdf2image)
from pdf2image import convert_from_path
images = convert_from_path(pdf_path, dpi=dpi)

# After (PyMuPDF)
import fitz
pdf_document = fitz.open(pdf_path)
page = pdf_document[page_num - 1]
mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
pix = page.get_pixmap(matrix=mat, alpha=False)
```

#### **문제 2: Candidate vs Consensus Barlines 혼동**
- **증상**: 모든 barline 후보들이 추출되어 과도한 마디 분할 발생
- **원인**: `filter_barlines()` 대신 `detect_barlines_per_system()` 사용 필요
- **해결책**: Multi-system consensus validation 적용
```python
# WRONG: Candidate barlines 사용
barlines = detector.filter_barlines(...)

# CORRECT: Consensus validation 사용  
validated_barlines = detector.detect_barlines_per_system(preprocessed)
```

### **Phase 2: GUI 통합 및 시각화**

#### **문제 3: System vs System Group 개념 혼동**
- **증상**: 잘못된 시스템에 barlines 적용
- **원인**: 개별 시스템(staff system)과 시스템 그룹(quartet grouping) 구분 실패
- **해결책**: 시스템 그룹별 barlines 매핑 구현
```python
# 시스템 그룹별로 barlines 분류
barlines_by_system_group = {}
for bl_info in barlines_with_systems:
    system_group_idx = bl_info.get('system_idx', 0)  # 시스템 그룹 인덱스
    # ...

# 각 그룹의 barlines을 그룹 내 모든 시스템에 적용
for group_idx, system_indices in enumerate(system_groups):
    group_barlines = barlines_by_system_group.get(group_idx, [])
    for sys_idx in system_indices:  # 그룹 내 모든 시스템
        # 마디 생성...
```

#### **문제 4: JSON Serialization 오류**
- **증상**: `Object of type int64 is not JSON serializable`
- **원인**: NumPy 데이터 타입이 JSON으로 변환되지 않음
- **해결책**: 모든 수치 데이터를 Python 기본 타입으로 변환
```python
# 모든 numpy 타입을 Python 타입으로 변환
"group_index": int(i),
"staff_lines": [{"y": int(y), "index": int(j)} for j, y in enumerate(...)],
"x": int(x), "y": int(y), "width": int(w), "height": int(h)
```

### **Phase 3: 최종 통합 및 검증**

#### **구현된 메타데이터 구조**
```json
{
  "page_number": 1,
  "page_dimensions": {"width": 2481, "height": 3508},
  "staff_groups": [
    {
      "group_index": 0,
      "staff_lines": [{"y": 553, "index": 0}, ...],
      "y_range": {"min": 553, "max": 634}
    }
  ],
  "system_clusters": [[0, 1, 2, 3], [4, 5, 6, 7], ...],
  "measures": [
    {
      "measure_id": "P1_00_001",
      "filename": "P1_00_001.png", 
      "staff_system_index": 0,
      "system_group_index": 0,
      "bounding_box_on_page": {"x": 0, "y": 533, "width": 385, "height": 121},
      "staff_line_coordinates_in_measure": [
        {"y": 20, "original_y": 553, "staff_index": 0, "group_index": 0}
      ]
    }
  ]
}
```

---

## 📊 구현 성과 및 품질 지표

### **정량적 성과**
- **테스트 파일**: `La Gazza ladra Overture` 1페이지 (2481×3508 픽셀)
- **검출된 시스템 그룹**: 3개 (각각 4개 시스템으로 구성된 4중주 편성)
- **추출된 마디 수**: 36개 (12개 마디 × 3개 시스템 그룹)
- **Consensus 검증율**: 80% 이상 합의 요구 (설정 가능)

### **품질 관리**
- **정확도**: GUI 시각화를 통한 실시간 검증 가능
- **일관성**: CLI와 GUI 동일한 결과 보장
- **추적성**: 모든 좌표 변환 과정이 메타데이터에 기록됨
- **확장성**: 다양한 악보 형식 (독주, 4중주, 오케스트라)에 대응

---

## 🗂️ 파일 구조 및 산출물

### **생성되는 디렉토리 구조**
```
output/measures/
├── metadata.json                    # 전체 프로젝트 메타데이터
└── page_01/
    ├── metadata.json               # 페이지별 상세 메타데이터
    ├── P1_00_001.png              # 시스템 0, 마디 1
    ├── P1_00_002.png              # 시스템 0, 마디 2
    ├── P1_01_001.png              # 시스템 1, 마디 1
    └── ...                        # 총 36개 마디 이미지
```

### **핵심 코드 모듈**
- **`extract_measures.py`**: 메인 CLI 도구 (311줄)
- **`scoreeye_gui.py`**: GUI 확장 (`generate_measure_boxes()`, `extract_measures()` 메서드 추가)
- **출력 파일 명명 규칙**: `P{page}_{system:02d}_{measure:03d}.png`

---

## 🎯 다음 단계 준비 완료

### **Phase 1.2 라벨링을 위한 기반 마련**
1. **고품질 마디 이미지**: 개별 마디별로 정제된 이미지 준비 완료
2. **정확한 메타데이터**: 오선 위치, 시스템 정보 등 라벨링에 필요한 모든 정보 제공
3. **배치 처리 능력**: 다양한 악보 PDF에 대한 대량 데이터셋 생성 가능
4. **품질 검증 도구**: GUI를 통한 실시간 검증 및 시각적 확인 가능

### **Roboflow 연동 준비**
- **이미지 포맷**: PNG 형식으로 표준화
- **메타데이터 호환성**: 향후 YOLO 학습에 필요한 모든 정보 포함
- **점진적 확장**: Level 1 클래스부터 시작 가능한 구조

---

## 🏁 결론

**Phase 1.1 마디 단위 이미지 및 메타데이터 생성** 단계가 성공적으로 완료되었습니다. 

- ✅ **CLI/GUI 듀얼 인터페이스** 구현으로 개발 및 프로덕션 환경 모두 지원
- ✅ **Consensus 검증 시스템** 적용으로 높은 정확도 확보  
- ✅ **시스템 그룹 인식** 구현으로 복잡한 앙상블 악보 처리 가능
- ✅ **완전한 메타데이터** 생성으로 다음 단계 YOLO 학습 준비 완료

이제 다음 단계인 **Phase 1.2 점진적 클래스 라벨링** 및 **Phase 2 YOLO 모델 학습**으로 진행할 수 있는 견고한 기반이 마련되었습니다.

---

**구현 완료일**: 2025년 7월 22일  
**다음 마일스톤**: Roboflow 프로젝트 설정 및 Level 1 클래스 라벨링 시작  
**예상 소요 시간**: Phase 1.2 완료까지 1주일