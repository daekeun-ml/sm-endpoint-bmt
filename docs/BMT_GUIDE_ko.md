# SageMaker Endpoint 벤치마크 도구 - 사용 가이드

CLI 는 `vllm bench serve` 와 같게 맞췄습니다. 거기서 되는 플래그는 여기서도 됩니다. 이 문서는 전체
인자를 `--help` 와 같은 그룹으로 나눠 정리합니다.

## 빠른 시작

```bash
# 가장 작은 실행: 측정 전에 endpoint 가 응답하는지부터 확인합니다
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --num-prompts 10

# 실제 측정, JSON 으로 저장
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --num-prompts 200 \
  --max-concurrency 20 \
  --save-result
```

## 명령줄 인자

아래 기본값은 argparse 기본값입니다. 기본값이 `None` 인 것은 **요청에서 그 필드를 아예 빼고** 컨테이너
기본값을 따른다는 뜻입니다.

### Endpoint

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--endpoint-name` | string | 필수 | SageMaker endpoint 이름 |
| `--region` | string | `us-east-1` | AWS 리전 |
| `--endpoint-url` | string | None | SageMaker Runtime API URL 재지정. 로컬 프록시로 대조 검증할 때 씁니다 |
| `--endpoint-type` / `--backend` | choice | `openai` | 페이로드 스키마: `openai`(`/v1/completions` 형식) 또는 `openai-chat`(messages). `completions`, `chat` 도 받습니다 |
| `--model` | string | None | 페이로드에 넣는 모델 id. 대부분의 LMI 컨테이너는 무시하므로, 한 컨테이너가 여러 모델을 서빙할 때만 지정합니다 |
| `--tokenizer` | string | None | HuggingFace tokenizer. random 프롬프트의 토큰 수를 정확히 맞추고, 출력 토큰 수 폴백으로도 씁니다 |
| `--label` | string | None | 결과 파일명에 들어가는 라벨 (미지정 시 `sagemaker`) |

### SageMaker 전송

AWS 경계가 평범한 HTTP 와 다르게 동작해서 필요한 값들입니다.

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--read-timeout` | int | 900 | botocore 소켓 읽기 타임아웃(초). botocore 기본값 60초는 긴 생성을 끊습니다 |
| `--connect-timeout` | int | 10 | botocore 연결 타임아웃(초) |
| `--no-include-usage` | flag | False | `stream_options.include_usage` 를 보내지 않습니다. 이 필드를 거부하는 컨테이너에만 씁니다. usage 가 없으면 출력 토큰 수가 추정치가 됩니다 |

### 데이터셋

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--dataset-name` | string | `random` | `random`, `sharegpt`, `huggingface`, `hf` |
| `--dataset-path` | string | None | 파일 경로(sharegpt) 또는 HuggingFace 데이터셋 ID |
| `--num-prompts` | int | 200 | 보낼 프롬프트 수 |
| `--disable-shuffle` | flag | False | 섞지 않고 순서대로 샘플링 |
| `--seed` | int | 0 | 샘플링 시드 |

### HuggingFace 데이터셋

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--hf-prompt-column` | string | `prompt` | 프롬프트가 있는 컬럼 |
| `--hf-completion-column` | string | `completion` | 답변이 있는 컬럼 |

### Random 데이터셋

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--random-input-len` | int | 1024 | 입력 길이(토큰) |
| `--random-output-len` | int | 128 | 출력 길이(토큰) |
| `--random-range-ratio` | float | 0.0 | 샘플링 범위: `len * (1 ± ratio)` |
| `--random-prefix-len` | int | 0 | 모든 프롬프트 앞에 붙는 고정 prefix 토큰 수 |

### 벤치마크

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--max-concurrency` | int | None | 동시 처리 상한. 미지정이면 상한이 없습니다 |
| `--request-rate` | float | `inf` | 초당 요청 수. `inf` 는 전부 한꺼번에 보냅니다 |
| `--burstiness` | float | 1.0 | 도착 과정의 gamma shape. `--request-rate` 가 유한할 때만 적용됩니다. 1.0 이 Poisson |
| `--num-warmups` | int | 0 | 워밍업 요청 수. 모든 지표에서 제외됩니다 |
| `--ready-check-timeout-sec` | int | 0 | endpoint 준비를 이만큼 기다립니다. 0 이면 검사를 건너뜁니다 |
| `--ignore-eos` | flag | False | `ignore_eos` 를 보내 요청한 길이까지 생성합니다. `--dataset-name random` 에서는 vLLM 과 같게 강제로 켜집니다 |
| `--disable-tqdm` | flag | False | 진행 표시줄 끄기 |
| `--percentile-metrics` | string | `ttft,tpot,itl` | 백분위를 낼 지표. 허용: `ttft`, `tpot`, `itl`, `e2el` |
| `--metric-percentiles` | string | `99` | 낼 백분위, 예: `25,50,75` |
| `--goodput` | KEY:VALUE… | None | 밀리초 단위 SLO, 예: `--goodput ttft:200 tpot:50` |
| `--ramp-up-strategy` | choice | None | `linear` 또는 `exponential`. 부하율을 일정하게 두지 않고 올립니다 |
| `--ramp-up-start-rps` | float | None | ramp 시작 rate |
| `--ramp-up-end-rps` | float | None | ramp 종료 rate |
| `--request-id-prefix` | string | 자동 생성 | 요청 id 접두사. 결과 JSON 에 기록됩니다 |

### 결과

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--save-result` | flag | False | 결과를 JSON 으로 저장 |
| `--save-detailed` | flag | False | 요청별 배열(ttfts, itls, errors)까지 포함 |
| `--append-result` | flag | False | 기존 파일에 JSONL 로 덧붙이기 |
| `--result-dir` | string | None | 출력 디렉터리 |
| `--result-filename` | string | None | 출력 파일명. 미지정이면 실행 파라미터로 생성합니다 |
| `--metadata` | KEY=VALUE… | None | JSON 에 함께 기록할 값, 예: `tp=4 engine=lmi` |

### 샘플링 파라미터

전부 기본값이 `None` 입니다. **요청에서 그 필드를 빼고** 서버 기본값을 따른다는 뜻이고,
`vllm bench serve` 와 같은 동작입니다.

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--temperature` | float | None | 지정하지 않으면 보내지 않습니다. greedy 를 원하면 `0` |
| `--top-p` | float | None | nucleus 샘플링 |
| `--top-k` | int | None | top-k 샘플링 |
| `--min-p` | float | None | 최소 확률 임계값 |
| `--repetition-penalty` | float | None | 1.0 초과면 반복을 억제 |
| `--presence-penalty` | float | None | 새 토큰을 장려 |
| `--frequency-penalty` | float | None | 자주 나온 토큰에 벌점 |
| `--extra-body` | JSON | None | 페이로드에 병합됩니다. 위 플래그보다 우선합니다 |
| `--use-beam-search` | flag | False | deprecated. 거부하는 컨테이너가 많습니다 |
| `--best-of` | int | None | deprecated. 거부하는 컨테이너가 많습니다 |

> **temperature 에 대해**: 이 도구는 더 이상 `temperature=0` 을 강제하지 않습니다. greedy 여부가 EOS
> 시점을 바꾸고, 그러면 출력 길이가 바뀌고, TPOT 과 처리량까지 달라집니다. 서버에 맡기는 것이 vLLM 의
> 동작이라 대조가 맞습니다. 예전처럼 하려면 `--temperature 0` 을 주세요.

## 사용 예시

### 1. 기본 random 데이터셋

```bash
uv run python sagemaker_benchmark.py --endpoint-name 'your-endpoint-name'
```

### 2. random 파라미터 조정

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --dataset-name random \
  --num-prompts 500 \
  --random-input-len 2048 \
  --random-output-len 256
```

### 3. chat 페이로드 스키마

컨테이너가 raw completions 대신 `/v1/chat/completions`(messages)를 서빙할 때 씁니다.

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --endpoint-type openai-chat \
  --num-prompts 100
```

### 4. ShareGPT 데이터셋

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --dataset-name sharegpt \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000
```

### 5. HuggingFace 데이터셋 (Alpaca)

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --dataset-name huggingface \
  --dataset-path "tatsu-lab/alpaca" \
  --hf-prompt-column "instruction" \
  --hf-completion-column "output" \
  --num-prompts 500
```

### 6. 고정 rate + 워밍업

워밍업이 첫 호출 비용을 흡수해서 백분위에 섞이지 않게 합니다.

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --request-rate 10 \
  --burstiness 1.0 \
  --num-warmups 5 \
  --num-prompts 500
```

### 7. 높은 동시성

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --num-prompts 1000 \
  --max-concurrency 50
```

### 8. ramp-up 으로 한계점 찾기

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --ramp-up-strategy linear \
  --ramp-up-start-rps 1 \
  --ramp-up-end-rps 20 \
  --num-prompts 600
```

### 9. SLO 기준 goodput

평균이 아니라, 모든 임계값을 만족한 요청 비율을 냅니다.

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --goodput ttft:200 tpot:50 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --num-prompts 300
```

### 10. 비교를 위한 결과 저장

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --save-result --save-detailed \
  --result-dir ./results \
  --label lmi26-tp4 \
  --metadata tp=4 engine=lmi \
  --num-prompts 200
```

### 11. 재현 가능한 생성

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --temperature 0 \
  --num-prompts 200
```

### 12. 컨테이너 고유 필드

플래그로 없는 옵션은 `--extra-body` 로 페이로드에 직접 병합합니다.

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --extra-body '{"stop": ["\n\n"], "seed": 42}' \
  --num-prompts 100
```

## 출력 지표

정의는 vLLM 과 동일합니다. `vllm bench serve` 결과와 나란히 놓고 비교할 수 있습니다.

### 요청 통계

- **Successful requests** / **Failed requests**
- **Maximum request concurrency**: `--max-concurrency` 상한값
- **Peak concurrent requests**: 실제로 관측된 최대 동시 요청 수
- **Request rate configured (RPS)**: 지시한 값이지 달성한 값이 아닙니다
- **Benchmark duration**

### 토큰 통계

- **Total input tokens** / **Total generated tokens**
- **Request throughput**(req/s), **Output token throughput**(tok/s), **Total token throughput**
- **Peak output token throughput**

### 지연 지표

| 지표 | 정의 |
|---|---|
| **TTFT** | 내용이 있는 첫 청크 도착 − 요청 전송 |
| **TPOT** | `(latency − ttft) / (output_len − 1)` |
| **ITL** | 연속한 청크 사이의 간격. TTFT 는 포함하지 않습니다 |
| **E2EL** | 요청 전송 → 마지막 청크 |

각 지표는 평균·중앙값·표준편차와 `--metric-percentiles` 로 지정한 백분위를 함께 냅니다.

### SageMaker Specifics

vLLM 출력에는 없는 절입니다. AWS 경계에서 생기는 것을 보고합니다.

- `finish_reason=length` 로 잘린 요청 수
- EOS 로 정상 종료한 요청 수
- 컨테이너가 usage 프레임을 보내지 않은 요청 수(토큰 수가 추정치로 대체됩니다)
- 예외 종류별 오류 분포

잘림은 오류가 아니라 결과입니다. 잘린 답변을 그냥 성공으로 세면 지연 수치가 틀어지므로 따로 뽑습니다.

## 팁

1. **`--num-prompts 10` 으로 시작**해서 측정 전에 endpoint 응답을 확인합니다.
2. **첫 실행을 손으로 버리지 말고 `--num-warmups`** 를 씁니다.
3. **`--max-concurrency` 를 실제 서비스할 값으로 두세요.** 없으면 상한이 없어서 순간 부하를 측정한
   숫자가 되고, 정상 상태를 나타내지 않습니다.
4. **`--request-rate` + `--burstiness`** 로 현실적인 도착 패턴을 만듭니다. rate 미지정은 전부 한꺼번에
   보내는 것입니다.
5. **실행 간 재현성이 필요하면 `--temperature 0`** 을 줍니다.
6. **잘림 수를 확인하세요.** `--dataset-name random` 에서 전부 잘리는 것은 정상입니다. 그 경우
   `--ignore-eos` 가 강제로 켜지기 때문입니다.

## 트러블슈팅

### 모든 요청이 실패

- endpoint 이름과 상태(`InService`)를 확인합니다
- AWS 자격증명과 `--region` 이 endpoint 가 있는 리전과 맞는지 확인합니다
- `uv run python test_endpoint.py` 를 먼저 돌립니다. 전송 문제와 측정 문제를 분리해 줍니다

### 실패율이 높음

- `--max-concurrency` 를 낮춥니다
- `--request-rate` 로 부하를 조절합니다
- CloudWatch 에서 throttling 과 `ModelLatency` 를 봅니다(단위가 **마이크로초**입니다)

### 지연이 예상보다 나쁨

- botocore 커넥션 풀은 `--max-concurrency` 에 맞춰 늘립니다. 직접 만든 클라이언트로 호출하면 기본값
  10 에 걸려 요청이 대기하고, 그러면 풀의 지연을 재게 됩니다
- 긴 생성이 끊기면 `--read-timeout` 을 올립니다

### 토큰 수가 이상함

- `--no-include-usage` 를 주면 출력 토큰 수가 추정치가 됩니다. 컨테이너가 usage 를 받으면 이 플래그를
  빼세요
- `--tokenizer` 에 모델 tokenizer 를 지정하면 입력 토큰 수가 정확해집니다
