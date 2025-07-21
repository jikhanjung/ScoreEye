# ScoreEye - 악보 마디 자동 인식 시스템

ScoreEye는 악보 이미지에서 마디(measure)를 자동으로 검출하고 개수를 세는 컴퓨터 비전 프로젝트입니다. 명령줄 도구와 GUI 애플리케이션을 모두 제공합니다.

## 주요 기능

- 악보 이미지 및 PDF 파일 지원
- 악보 이미지에서 오선(staff lines) 자동 검출
- 세로줄(barlines) 검출 및 필터링
- 마디 개수 자동 계산
- 검출 결과 시각화

## 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd ScoreEye
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv

# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

### GUI 애플리케이션

#### 실행
```bash
python scoreeye_gui.py
```

#### GUI 기능
- PDF 파일 열기 및 페이지 탐색
- 실시간 검출 결과 오버레이
- 줌 인/아웃 기능
- 오선 및 바라인 표시 토글
- 검출 결과 이미지 내보내기
- DPI 조정 가능

### 명령줄 도구

#### 이미지 파일 처리
```bash
python detect_measure.py images/sample_score.png
```

#### PDF 파일 처리
```bash
python detect_measure.py pdfs/score.pdf
```

### 옵션
- `-o, --output`: 결과 이미지 저장 경로 지정
- `-d, --debug`: 디버그 모드 (중간 처리 단계 시각화)
- `-p, --page`: PDF 파일의 특정 페이지 지정 (기본값: 1)
- `--dpi`: PDF 변환 해상도 지정 (기본값: 300)

### 예제
```bash
# 이미지 파일 처리
python detect_measure.py images/score.png -o output/result.png

# PDF 파일의 첫 페이지 처리
python detect_measure.py pdfs/score.pdf

# PDF 파일의 3번째 페이지를 고해상도로 처리
python detect_measure.py pdfs/score.pdf -p 3 --dpi 600

# 디버그 모드로 실행
python detect_measure.py images/score.png -d
```

## 프로젝트 구조

```
ScoreEye/
├── images/           # 입력 악보 이미지
├── pdfs/            # 입력 PDF 파일
├── output/          # 처리 결과 이미지
├── detect_measure.py # 메인 검출 스크립트 (CLI)
├── scoreeye_gui.py  # GUI 애플리케이션
├── requirements.txt # Python 의존성
├── README.md       # 프로젝트 문서
└── CLAUDE.md       # Claude Code 가이드
```

## 기술적 세부사항

### 처리 과정
1. **이미지 전처리**: Otsu 이진화 및 노이즈 제거
2. **오선 검출**: 수평 투영(horizontal projection) 분석
3. **바라인 검출**: 형태학적 연산을 통한 수직선 추출
4. **바라인 필터링**: 오선을 완전히 교차하는 선만 선택
5. **마디 계산**: 검증된 바라인 개수로 마디 수 산출

### 주요 알고리즘
- 이진화: Otsu's method
- 오선 검출: 수평 투영 + 피크 검출
- 바라인 검출: Morphological opening with vertical kernel
- 필터링: 오선 교차 검증

## 제한사항

- PDF의 경우 한 번에 한 페이지만 처리 가능
- 너무 작거나 해상도가 낮은 이미지는 정확도가 떨어질 수 있음
- 복잡한 악보 기호(반복 기호 등)는 추가 처리 필요
- PDF 처리 시 poppler-utils 설치 필요 (pdf2image 의존성)

## 향후 개선 계획

- [ ] 다중 페이지 악보 지원
- [ ] 반복 기호 및 특수 바라인 인식
- [ ] 마디별 이미지 분할 기능
- [ ] 딥러닝 기반 검출 정확도 향상

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.