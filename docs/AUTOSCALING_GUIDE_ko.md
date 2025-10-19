# Auto Scaling 가이드

SageMaker Endpoint Auto Scaling 설정 및 사용 가이드입니다.

## 목차

- [개요](#개요)
- [CloudWatch 메트릭](#cloudwatch-메트릭)
- [Auto Scaling 설정](#auto-scaling-설정)
- [테스트 시나리오](#테스트-시나리오)
- [모니터링](#모니터링)
- [Best Practices](#best-practices)
- [문제 해결](#문제-해결)

---

## 개요

SageMaker Endpoint Auto Scaling은 CloudWatch 메트릭을 기반으로 자동으로 인스턴스 수를 조정합니다.

### 주요 기능

- **자동 확장 (Scale Out)**: 트래픽 증가 시 인스턴스 추가
- **자동 축소 (Scale In)**: 트래픽 감소 시 인스턴스 제거
- **메트릭 기반**: CloudWatch 메트릭으로 스케일링 결정
- **Cooldown 기간**: 급격한 스케일링 방지

### 동작 원리

```
High Traffic → CloudWatch Metric ↑ → Scale Out → More Instances
Low Traffic  → CloudWatch Metric ↓ → Scale In  → Fewer Instances
```

---

## CloudWatch 메트릭

### 1. InvocationsPerInstance (Predefined Metric)

**타입:** Predefined Metric  
**설명:** 인스턴스당 요청 수 (분당 평균)

**사용 시기:**
- 일반적인 스케일링
- 요청 수 기반 확장
- 안정적인 트래픽 패턴

**권장 값:**
- 소형 모델 (< 10B): 1000-2000
- 중형 모델 (10B-70B): 500-1000
- 대형 모델 (> 70B): 100-500

**예시:**
```bash
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --target-invocations-per-instance 1000
```

### 2. ConcurrentRequestsPerModel (High-Resolution Predefined Metric)

**타입:** High-Resolution Predefined Metric (10초 간격)  
**설명:** 모델당 동시 요청 수

**주요 특징:**
- 10초 간격으로 수집되는 고해상도 메트릭
- 최대 6배 빠른 스케일 업 감지
- Generative AI 모델에 최적화
- 1분 미만의 빠른 스케일링

**사용 시기:**
- 동시성이 중요한 경우
- 실시간 응답이 필요한 경우
- 트래픽 급증에 빠르게 대응
- LLM 및 Foundation Model

**권장 값:**
- 소형 모델: 50-100
- 중형 모델: 20-50
- 대형 모델: 5-20

**예시:**
```bash
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --target-concurrent-requests-per-model 50
```

**참고:**
- AWS Blog: [Faster Auto Scaling for Generative AI Models](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-inference-launches-faster-auto-scaling-for-generative-ai-models/)
- 가속기 인스턴스 (g4dn, g5, g6, p2, p3, p4d, p4de, p5, inf1, inf2, trn1n, trn1)에서 사용 가능

### 3. ConcurrentRequestsPerCopy (High-Resolution Predefined Metric)

**타입:** High-Resolution Predefined Metric (10초 간격)  
**설명:** Inference Component의 각 Copy(복사본)당 동시 요청 수

**Inference Components란?**
- 하나의 Endpoint에 여러 모델을 배포하는 기능
- 각 모델은 독립적인 Inference Component로 배포
- 리소스(CPU, GPU, 메모리)를 세밀하게 할당 가능

**Copy란?**
- 모델 컨테이너의 런타임 복사본
- `CopyCount` 파라미터로 개수 지정
- 각 Copy는 독립적으로 추론 요청 처리

**예시 구조:**
```
Endpoint
├── Inference Component A (Llama-70B)
│   ├── Copy 1 (GPU 0-3)
│   └── Copy 2 (GPU 4-7)
└── Inference Component B (Llama-8B)
    ├── Copy 1 (GPU 0)
    ├── Copy 2 (GPU 1)
    └── Copy 3 (GPU 2)
```

**주요 특징:**
- 10초 간격으로 수집되는 고해상도 메트릭
- Inference Components의 Copy 수를 자동 조정
- 여러 모델을 효율적으로 호스팅

**사용 시기:**
- Inference Components 사용 시 (필수)
- 하나의 Endpoint에 여러 모델 배포
- 모델별 독립적인 스케일링 필요
- 리소스 효율성 최적화

**권장 값:**
- 일반적으로 5-20
- 모델 크기와 처리 시간에 따라 조정

**예시:**
```bash
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --target-concurrent-requests-per-copy 10
```

**참고:**
- Inference Components를 사용하지 않는 일반 Endpoint에서는 이 메트릭을 사용하지 않습니다
- 대신 `ConcurrentRequestsPerModel` 사용
- [Inference Components 문서](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html)

### 메트릭 비교

| 메트릭 | 타입 | 수집 간격 | 사용 사례 | 스케일링 속도 |
|--------|------|-----------|-----------|---------------|
| InvocationsPerInstance | Predefined | 1분 | 일반 Endpoint (단일 모델) | 표준 |
| ConcurrentRequestsPerModel | High-Resolution | 10초 | 일반 Endpoint (LLM, 실시간) | 최대 6배 빠름 |
| ConcurrentRequestsPerCopy | High-Resolution | 10초 | Inference Components (다중 모델) | 최대 6배 빠름 |

### 하이브리드 접근 방식 (권장)

여러 메트릭을 동시에 사용하여 더 효율적인 스케일링을 구현할 수 있습니다:

```bash
# InvocationsPerInstance + ConcurrentRequestsPerModel 조합
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 1 \
  --max-capacity 10 \
  --target-invocations-per-instance 1000 \
  --target-concurrent-requests-per-model 50
```

**장점:**
- 급격한 트래픽 증가: ConcurrentRequestsPerModel이 빠르게 대응
- 점진적 트래픽 증가: InvocationsPerInstance가 안정적으로 대응
- 컨테이너 장애: 두 메트릭이 상호 보완

---

## Auto Scaling 설정

### 기본 설정

```bash
# 1. Auto Scaling 설정
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 1 \
  --max-capacity 10 \
  --target-invocations-per-instance 1000

# 2. 설정 확인
uv run python autoscaling/autoscaling.py describe --endpoint-name "my-endpoint"

# 3. 메트릭 확인
uv run python autoscaling/autoscaling.py metrics --endpoint-name "my-endpoint"
```

### 고급 설정 (여러 메트릭 동시 사용)

```bash
# 3개 메트릭 모두 사용 (하이브리드 접근)
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 2 \
  --max-capacity 20 \
  --target-invocations-per-instance 1000 \
  --target-concurrent-requests-per-model 50 \
  --target-concurrent-requests-per-copy 10 \
  --scale-in-cooldown 300 \
  --scale-out-cooldown 60
```

**동작 방식:**
- 3개의 독립적인 Auto Scaling 정책이 생성됨
- 각 정책이 독립적으로 메트릭을 모니터링
- 어느 정책이든 임계값 초과 시 Scale Out 트리거
- 모든 정책의 메트릭이 목표값 이하일 때 Scale In

**권장 조합:**
1. **LLM 워크로드:**
   ```bash
   --target-invocations-per-instance 500 \
   --target-concurrent-requests-per-model 30
   ```

2. **실시간 추론:**
   ```bash
   --target-concurrent-requests-per-model 50 \
   --target-concurrent-requests-per-copy 10
   ```

3. **안정적인 워크로드:**
   ```bash
   --target-invocations-per-instance 1000
   ```

### 파라미터 설명

| 파라미터 | 설명 | 기본값 | 권장값 |
|---------|------|--------|--------|
| `min-capacity` | 최소 인스턴스 수 | 1 | 1-2 |
| `max-capacity` | 최대 인스턴스 수 | 10 | 예상 최대 부하의 1.5배 |
| `scale-in-cooldown` | Scale In 대기 시간 (초) | 300 | 300-600 |
| `scale-out-cooldown` | Scale Out 대기 시간 (초) | 60 | 30-60 |

---

## 테스트 시나리오

### 1. Full Cycle 테스트 (권장)

전체 Auto Scaling 사이클을 테스트합니다.

```bash
uv run python autoscaling/test_autoscaling.py \
  --endpoint-name "my-endpoint" \
  --test full-cycle \
  --num-requests 100 \
  --duration 300 \
  --scale-in-wait 600 \
  --metrics-interval 30
```

**테스트 흐름:**
1. 초기 상태 확인
2. 높은 부하 발생 (5분)
3. Scale Out 확인
4. 부하 중단
5. Scale In 확인

**예상 결과:**
```
Initial Instances: 1
After Scale Out: 3-5
After Scale In: 1
```

### 2. Scale Out 테스트

Scale Out만 테스트합니다.

```bash
uv run python autoscaling/test_autoscaling.py \
  --endpoint-name "my-endpoint" \
  --test scale-out \
  --num-requests 100 \
  --duration 300
```

**테스트 흐름:**
1. 초기 상태 확인
2. 높은 부하 발생
3. CloudWatch 메트릭 모니터링
4. 인스턴스 증가 확인

**성공 기준:**
- 메트릭이 목표값 초과
- 인스턴스 수 증가
- 요청 처리 성공

### 3. Scale In 테스트

Scale In만 테스트합니다.

```bash
uv run python autoscaling/test_autoscaling.py \
  --endpoint-name "my-endpoint" \
  --test scale-in \
  --scale-in-wait 600
```

**테스트 흐름:**
1. 현재 상태 확인 (여러 인스턴스)
2. 부하 없이 대기
3. Cooldown 기간 경과
4. 인스턴스 감소 확인

**성공 기준:**
- 메트릭이 목표값 미만
- Cooldown 기간 경과 후 인스턴스 감소

---

## 모니터링

### CloudWatch 메트릭 확인

```bash
# 최근 5분 메트릭
uv run python autoscaling/autoscaling.py metrics --endpoint-name "my-endpoint"

# 최근 30분 메트릭
uv run python autoscaling/autoscaling.py metrics --endpoint-name "my-endpoint" --period 30
```

### AWS Console에서 확인

1. **CloudWatch 대시보드**
   ```
   https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:
   ```

2. **SageMaker Endpoint 모니터링**
   ```
   https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/endpoints
   ```

3. **Auto Scaling 정책**
   ```
   https://console.aws.amazon.com/ec2autoscaling/home?region=us-east-1#/details
   ```

### 주요 확인 사항

- **Current Instance Count**: 현재 인스턴스 수
- **Desired Instance Count**: 목표 인스턴스 수
- **InvocationsPerInstance**: 인스턴스당 요청 수
- **ConcurrentRequests**: 동시 요청 수
- **Model Latency**: 모델 지연 시간

---

## Best Practices

### 1. 초기 설정

```bash
# 보수적인 시작
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 1 \
  --max-capacity 5 \
  --target-invocations-per-instance 500 \
  --scale-in-cooldown 600 \
  --scale-out-cooldown 60
```

**이유:**
- 낮은 목표값으로 빠른 확장
- 긴 Scale In Cooldown으로 안정성 확보
- 작은 max-capacity로 비용 제어

### 2. 프로덕션 설정

```bash
# 실제 트래픽 패턴 분석 후
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 2 \
  --max-capacity 20 \
  --target-invocations-per-instance 1000 \
  --scale-in-cooldown 300 \
  --scale-out-cooldown 60
```

**이유:**
- min-capacity 2로 고가용성 확보
- 실제 트래픽 기반 목표값 설정
- 적절한 max-capacity로 확장성 확보

### 3. 비용 최적화

```bash
# 비용 중심
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 1 \
  --max-capacity 10 \
  --target-invocations-per-instance 1500 \
  --scale-in-cooldown 180 \
  --scale-out-cooldown 60
```

**이유:**
- 높은 목표값으로 인스턴스 수 최소화
- 짧은 Scale In Cooldown으로 빠른 축소
- min-capacity 1로 유휴 시간 비용 절감

### 4. 성능 최적화

```bash
# 성능 중심
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 3 \
  --max-capacity 30 \
  --target-invocations-per-instance 500 \
  --target-concurrent-requests-per-model 30 \
  --scale-in-cooldown 600 \
  --scale-out-cooldown 30
```

**이유:**
- 낮은 목표값으로 여유 확보
- 빠른 Scale Out으로 지연 최소화
- 여러 메트릭으로 세밀한 제어

---

## 문제 해결

### Scale Out이 느림

**증상:**
- 트래픽 증가해도 인스턴스가 늘어나지 않음
- 응답 시간이 길어짐

**해결 방법:**
1. 목표값 낮추기
   ```bash
   --target-invocations-per-instance 500  # 1000에서 낮춤
   ```

2. Scale Out Cooldown 줄이기
   ```bash
   --scale-out-cooldown 30  # 60에서 줄임
   ```

3. 메트릭 확인
   ```bash
   uv run python autoscaling/autoscaling.py metrics --endpoint-name "my-endpoint"
   ```

### Scale In이 너무 빠름

**증상:**
- 트래픽이 조금만 줄어도 인스턴스 감소
- 다시 트래픽 증가 시 지연 발생

**해결 방법:**
1. Scale In Cooldown 늘리기
   ```bash
   --scale-in-cooldown 600  # 300에서 늘림
   ```

2. 목표값 조정
   ```bash
   --target-invocations-per-instance 800  # 여유 확보
   ```

### 인스턴스가 max-capacity에 도달

**증상:**
- 계속 max-capacity 유지
- 응답 시간이 여전히 김

**해결 방법:**
1. max-capacity 증가
   ```bash
   --max-capacity 20  # 10에서 증가
   ```

2. 목표값 재검토
   ```bash
   # 현재 메트릭 확인
   uv run python autoscaling/autoscaling.py metrics --endpoint-name "my-endpoint"
   
   # 적절한 목표값으로 조정
   --target-invocations-per-instance 1200
   ```

3. 인스턴스 타입 업그레이드
   ```bash
   # 더 큰 인스턴스 사용
   --instance-type "ml.g5.12xlarge"
   ```

### CloudWatch 메트릭이 보이지 않음

**증상:**
- `uv run python autoscaling/autoscaling.py metrics` 실행 시 데이터 없음

**해결 방법:**
1. Endpoint에 요청 전송
   ```bash
   python test_endpoint.py --endpoint-name "my-endpoint"
   ```

2. 시간 대기 (1-2분)

3. 다시 확인
   ```bash
   uv run python autoscaling/autoscaling.py metrics --endpoint-name "my-endpoint"
   ```

### Auto Scaling 정책이 적용되지 않음

**증상:**
- 설정했지만 스케일링이 안 됨

**해결 방법:**
1. 설정 확인
   ```bash
   uv run python autoscaling/autoscaling.py describe --endpoint-name "my-endpoint"
   ```

2. IAM 권한 확인
   - `application-autoscaling:*` 권한 필요
   - `cloudwatch:GetMetricStatistics` 권한 필요

3. 재설정
   ```bash
   # 삭제 후 재설정
   uv run python autoscaling/autoscaling.py delete --endpoint-name "my-endpoint"
   uv run python autoscaling/autoscaling.py configure --endpoint-name "my-endpoint" ...
   ```

---

## 참고 자료

- [AWS SageMaker Auto Scaling](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html)
- [CloudWatch Metrics for SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html)
- [Application Auto Scaling](https://docs.aws.amazon.com/autoscaling/application/userguide/what-is-application-auto-scaling.html)
