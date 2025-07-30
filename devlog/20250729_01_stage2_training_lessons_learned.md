# Stage 2 Training Lessons Learned and Preprocessing Improvements

작성일: 2025-07-29  
작성자: ScoreEye Development Team

## 1. Executive Summary

Stage 2 YOLOv8 훈련 중 발견된 중요한 문제점들과 개선사항을 정리합니다. 주요 발견은 클래스 ID 매핑 오류로 인한 clefG(높은음자리표) 누락과 제외하려던 어려운 클래스들의 포함입니다.

## 2. 현재 Stage 2 훈련 상태

### 2.1 훈련 구성
- **모델**: YOLOv8s
- **배치 크기**: 1 (메모리 안정성 우선)
- **이미지 크기**: 1024×1024
- **학습률**: 0.001 (Stage 1의 0.01에서 감소)
- **옵티마이저**: SGD
- **에폭**: 100 (현재 66/100 진행 중)

### 2.2 성능 지표 (66 에폭 기준)
- **mAP@50**: 0.75
- **mAP@50-95**: 0.58
- **Precision**: 0.908
- **Recall**: 0.661

### 2.3 메모리 사용량
- **평균**: 3.9GB / 11GB (35% 사용률)
- **최대**: 6.76GB (안정적)
- **배치 2 시도 시**: 간헐적으로 11GB 초과 (에폭당 4시간 소요)

## 3. 발견된 문제점

### 3.1 치명적인 클래스 누락: clefG
```python
# Stage 2 전처리 스크립트의 제외 클래스
self.excluded_classes = {
    '42',   # stem ✓
    '2',    # ledgerLine ✓
    '6',    # beam으로 의도했으나 실제로는 clefG! ❌
    '68',   # tie ✓
    '54'    # slur ✓
}
```

**문제**: ID 6은 beam이 아니라 clefG(높은음자리표)였음
- clefG는 12,919개 샘플을 가진 12번째로 많은 클래스
- 악보 인식에서 가장 중요한 기호 중 하나
- 이로 인해 전체 시스템의 인식 능력 저하

### 3.2 제외하려던 클래스들이 포함됨
YAML 파일을 확인한 결과, 제외하려던 클래스들이 실제로는 포함되어 있었습니다:
- slur (43번)
- beam (44번)
- tie (45번)

이들은 얇고 긴 형태로 YOLOv8로 검출하기 어려운 클래스들입니다.

### 3.3 클래스 매핑 불일치
`excluded_classes_info.json`의 클래스 이름이 "Unknown_"으로 표시되어 실제 제외된 클래스를 파악하기 어려웠습니다.

## 4. 메모리 사용 패턴 분석

### 4.1 배치 크기별 특성
**배치 1**:
- 안정적이지만 GPU 활용도 낮음 (35%)
- 에폭당 32분
- 메모리 스파이크 없음

**배치 2**:
- 평균적으로 빠름 (에폭당 20-22분)
- 간헐적 메모리 초과로 극도로 느려짐 (4시간/에폭)
- GPU 활용도 70%

### 4.2 모델 크기별 예상
**YOLOv8s (현재)**:
- 11.2M 파라미터
- 최대 6.76GB 메모리 사용

**YOLOv8m (다음 계획)**:
- 25.9M 파라미터 (2.3배)
- 예상 최대 10-11GB (1.7배 증가 예상)
- 배치 1로만 안정적 훈련 가능

## 5. 다음 Stage 2 전처리 개선사항

### 5.1 올바른 클래스 선택
```python
# 수정된 제외 클래스 (다음 전처리 시)
self.excluded_classes = {
    '42',   # stem
    '2',    # ledgerLine  
    # beam의 실제 ID 찾아서 추가 (6이 아님!)
    # tie의 실제 ID 확인
    # slur의 실제 ID 확인
}

# 반드시 포함해야 할 클래스
# ID 6 = clefG (높은음자리표) - 필수!
```

### 5.2 클래스 ID 검증 단계 추가
```python
def verify_class_exclusions(self):
    """제외하려는 클래스가 올바른지 이름으로 검증"""
    for class_id in self.excluded_classes:
        class_name = self.id_to_name.get(class_id, "Unknown")
        print(f"제외 예정: ID {class_id} = {class_name}")
        
    # 중요 클래스가 제외되지 않았는지 확인
    critical_classes = ['clefG', 'clefF', 'noteheadBlack*', ...]
    for critical in critical_classes:
        if critical in excluded_names:
            raise ValueError(f"중요 클래스 {critical}가 제외 목록에 있습니다!")
```

### 5.3 전처리 설정 권장사항
```json
{
    "model": "yolov8m.pt",
    "batch_size": 1,  // 메모리 안정성 최우선
    "image_size": 1024,
    "learning_rate": 0.001,
    "optimizer": "SGD",
    "epochs": 100,
    "patience": 20,
    "mosaic": 0.0,  // 메모리 절약
    "warmup_epochs": 5
}
```

## 6. 학습된 교훈

### 6.1 클래스 ID는 반드시 이름으로 검증
- 숫자 ID만으로는 실수하기 쉬움
- 전처리 전 반드시 이름 매핑 확인

### 6.2 메모리 관리의 중요성
- RTX 2080 Ti (11GB)에서는 보수적 설정 필요
- 간헐적 메모리 스파이크를 고려한 여유 확보
- 배치 1이 느려도 안정성이 더 중요

### 6.3 단계적 접근의 장점
- Stage 1 실패 → Stage 2로 복잡도 감소
- 각 단계에서 얻은 교훈을 다음에 적용
- 완주가 최적화보다 중요

## 7. YOLOv8m 재훈련 계획

### 7.1 전처리 체크리스트
- [ ] beam, tie, slur의 정확한 ID 파악
- [ ] clefG(ID: 6) 반드시 포함 확인
- [ ] 클래스 이름으로 최종 검증
- [ ] 총 45개 클래스 확인

### 7.2 훈련 전략
1. **안정성 우선**: 배치 1, mosaic 0.0
2. **충분한 훈련**: 100 에폭
3. **조기 종료**: patience 20
4. **모니터링**: 메모리 사용량 추적

### 7.3 예상 결과
- mAP@50: 0.82-0.85 (현재 0.75에서 향상)
- 훈련 시간: 80-90시간
- 메모리 사용: 평균 6-7GB, 최대 10-11GB

## 8. 결론

Stage 2 훈련은 여러 문제에도 불구하고 YOLOv8 파이프라인의 안정성을 검증했습니다. 클래스 ID 매핑 오류는 치명적이었지만, 이를 통해 전처리의 중요성을 재확인했습니다. YOLOv8m으로의 재훈련 시 이러한 교훈들을 적용하면 더 나은 결과를 얻을 수 있을 것입니다.

---

**다음 단계**: 
1. 현재 YOLOv8s 훈련 완료 (약 17시간 남음)
2. 결과 분석 및 클래스별 성능 확인
3. 전처리 스크립트 수정
4. YOLOv8m으로 재훈련 시작