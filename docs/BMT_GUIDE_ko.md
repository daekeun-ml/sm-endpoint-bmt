# SageMaker Endpoint 벤치마크 도구 - 사용 가이드

## 빠른 시작

```bash
# 랜덤 데이터셋으로 기본 벤치마크
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name'

# 커스텀 파라미터 사용
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --num-prompts 500 \
  --max-concurrency 20 \
  --temperature 0.7
```

## 명령줄 인자

### 필수 인자

| 인자 | 타입 | 설명 |
|------|------|------|
| `--endpoint-name` | string | SageMaker endpoint 이름 (필수) |

### Endpoint 인자

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--region` | string | None | AWS 리전 (지정하지 않으면 기본 리전 사용) |

### 데이터셋 인자

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--dataset-name` | string | `random` | 데이터셋 타입: `random`, `sharegpt`, `huggingface`, `hf` |
| `--dataset-path` | string | None | 데이터셋 파일 경로 (sharegpt) 또는 HuggingFace dataset ID (huggingface) |
| `--num-prompts` | int | 200 | 처리할 프롬프트 수 |
| `--disable-shuffle` | flag | False | 데이터셋 셔플 비활성화 |
| `--seed` | int | 0 | 데이터셋 샘플링을 위한 랜덤 시드 |

### HuggingFace 데이터셋 인자

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--hf-prompt-column` | string | `prompt` | HuggingFace 데이터셋의 프롬프트 컬럼명 |
| `--hf-completion-column` | string | `completion` | HuggingFace 데이터셋의 완성 컬럼명 |

### Random 데이터셋 인자

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--random-input-len` | int | 1024 | 랜덤 입력 길이 (토큰 단위) |
| `--random-output-len` | int | 128 | 랜덤 출력 길이 (토큰 단위) |

### 벤치마크 인자

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--max-concurrency` | int | 10 | 최대 동시 요청 수 |
| `--request-rate` | string | `inf` | 초당 요청 속도. `inf`는 무제한 |

### 샘플링 파라미터

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--temperature` | float | 0.0 | 샘플링 온도 (0.0 = greedy, 높을수록 랜덤) |
| `--top-p` | float | 1.0 | Top-p (nucleus) 샘플링 파라미터 (0.0 ~ 1.0) |
| `--top-k` | int | -1 | Top-k 샘플링 파라미터 (-1 = 비활성화) |
| `--use-beam-search` | flag | False | 샘플링 대신 beam search 사용 |
| `--best-of` | int | 1 | 생성할 시퀀스 수 (가장 좋은 것 선택) |
| `--repetition-penalty` | float | 1.0 | 반복 페널티 (1.0 = 페널티 없음, >1.0 = 반복 억제) |
| `--presence-penalty` | float | 0.0 | Presence 페널티 (새로운 토큰 생성 장려) |
| `--frequency-penalty` | float | 0.0 | Frequency 페널티 (빈번한 토큰 억제) |

## 사용 예시

### 1. 기본 랜덤 데이터셋

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438'
```

### 2. 커스텀 랜덤 데이터셋 파라미터

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --dataset-name random \
  --num-prompts 500 \
  --random-input-len 2048 \
  --random-output-len 256
```

### 3. ShareGPT 데이터셋

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --dataset-name sharegpt \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000
```

### 4. HuggingFace 데이터셋 (Alpaca)

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --dataset-name huggingface \
  --dataset-path "tatsu-lab/alpaca" \
  --hf-prompt-column "instruction" \
  --hf-completion-column "output" \
  --num-prompts 500
```

### 5. 높은 동시성 테스트

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --num-prompts 1000 \
  --max-concurrency 50
```

### 6. 속도 제한 요청

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --request-rate 10 \
  --num-prompts 500
```

### 7. 창의적 생성 (높은 Temperature)

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --temperature 0.8 \
  --top-p 0.95 \
  --num-prompts 200
```

### 8. 결정적 생성 (Greedy Decoding)

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --temperature 0.0 \
  --num-prompts 200
```

### 9. Top-k 샘플링과 반복 페널티

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --temperature 0.7 \
  --top-k 50 \
  --repetition-penalty 1.1 \
  --num-prompts 200
```

### 10. Beam Search

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --use-beam-search \
  --best-of 3 \
  --num-prompts 100
```

### 11. 셔플 비활성화 (순차 샘플링)

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --dataset-name sharegpt \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --disable-shuffle \
  --num-prompts 100
```

## 출력 메트릭

벤치마크 도구는 다음 메트릭을 제공합니다:

### 요청 통계
- **Successful requests**: 성공적으로 완료된 요청 수
- **Failed requests**: 실패한 요청 수
- **Maximum request concurrency**: 벤치마크 중 최대 동시 요청 수
- **Benchmark duration**: 벤치마크 총 소요 시간

### 토큰 통계
- **Total input tokens**: 처리된 총 입력 토큰 수
- **Total generated tokens**: 생성된 총 출력 토큰 수
- **Request throughput**: 초당 요청 처리량
- **Output token throughput**: 초당 출력 토큰 처리량
- **Total token throughput**: 초당 총 토큰 (입력 + 출력) 처리량

### 지연 시간 메트릭
- **TTFT (Time to First Token)**: 요청 시작부터 첫 토큰까지의 시간
  - 평균, 중앙값, P99 (밀리초 단위)
- **TPOT (Time per Output Token)**: 출력 토큰당 평균 시간 (첫 토큰 제외)
  - 평균, 중앙값, P99 (밀리초 단위)
- **ITL (Inter-token Latency)**: 연속된 토큰 간 시간
  - 평균, 중앙값, P99 (밀리초 단위)

## 팁

1. **적은 수의 프롬프트로 시작**하여 endpoint가 정상 작동하는지 확인하세요
2. **재현 가능한 결과**를 위해 `--temperature 0.0` 사용
3. **endpoint 용량에 맞춰** `--max-concurrency` 조정
4. **실제 트래픽 패턴 시뮬레이션**을 위해 `--request-rate` 사용
5. **실패한 요청 모니터링** - 많은 실패가 보이면 동시성이나 요청 속도 감소

## 문제 해결

### 모든 요청이 실패하는 경우
- Endpoint 이름이 올바른지 확인
- AWS 자격 증명이 설정되어 있는지 확인
- Endpoint가 올바른 리전에 있는지 확인

### 높은 실패율
- `--max-concurrency` 감소
- `--request-rate`를 추가하여 요청 속도 제한
- Endpoint CloudWatch 메트릭에서 throttling 확인

### 예상치 못한 메트릭
- Endpoint 응답 형식이 OpenAI completion API와 일치하는지 확인
- 먼저 테스트 스크립트 실행: `uv run python test_endpoint.py`
