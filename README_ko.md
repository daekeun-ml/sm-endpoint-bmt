# SageMaker Endpoint Benchmark Tool

vLLM bench serve와 유사한 SageMaker endpoint 벤치마크 도구입니다.

## 설치

### uv 사용 (권장)

```bash
# uv 설치 (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 의존성 설치 및 가상환경 생성
uv sync

# 스크립트 실행 (uv가 자동으로 가상환경 활성화)
uv run python sagemaker_benchmark.py --help
```

### pip 사용 (대안)

```bash
# 가상환경 생성 및 활성화 (권장)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 또는 venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

> **권장사항**: uv는 pip보다 10-100배 빠르고 더 안정적인 Python 패키지 관리자입니다. 자동으로 가상환경을 관리하고 의존성 해결이 뛰어납니다. 자세한 내용은 [uv 공식 문서](https://docs.astral.sh/uv/)를 참조하세요.

## 설정

### 1. 서빙 설정 파일

설정 파일은 `config/<엔진>/` 아래에 있습니다. 예시를 복사해 `.example` 를 떼면 됩니다:

```bash
# vLLM DLC (기본)
cp config/vllm/gemma-4-E4B.json.example config/vllm/gemma-4-E4B.json

# DJL LMI
cp config/lmi/gpt-oss-20b.json.example config/lmi/gpt-oss-20b.json
```

두 엔진에 두 모델군을 모두 넣었습니다. gemma-4 는 5종, gpt-oss 는 2종입니다. 전체 목록은
[Config 폴더 구조](#config-폴더-구조)에 있습니다.

`config/vllm/gemma-4-E4B.json` (`SM_VLLM_*` 키):
```json
{
  "SM_VLLM_MODEL": "google/gemma-4-E4B-it",
  "SM_VLLM_TENSOR_PARALLEL_SIZE": "1",
  "SM_VLLM_MAX_MODEL_LEN": "4096",
  "SM_VLLM_MAX_NUM_SEQS": "32",
  "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.90",
  "HF_TOKEN": ""
}
```

`config/lmi/gpt-oss-20b.json` (`OPTION_*` 키):
```json
{
  "HF_MODEL_ID": "openai/gpt-oss-20b",
  "HF_TOKEN": "",
  "OPTION_MODEL_LOADING_TIMEOUT": "1500",
  "SERVING_FAIL_FAST": "true",
  "OPTION_ASYNC_MODE": "true",
  "OPTION_ROLLING_BATCH": "disable",
  "TENSOR_PARALLEL_DEGREE": "max",
  "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service"
}
```

`create_endpoint.py` 가 env 키를 읽어 어느 컨테이너를 띄울지 판단하므로 엔진을 명령줄로 넘길 일은
없습니다. `_` 로 시작하는 키는 값의 근거를 적어 둔 주석이고, 컨테이너에 전달되기 전에 제거됩니다.

### 2. 환경 변수 설정 (선택사항)

`.env.example` 파일을 복사하여 `.env` 파일을 만들고 인스턴스 설정을 입력합니다:

```bash
cp .env.example .env
```

`.env` 파일 예시:
```bash
INSTANCE_TYPE=ml.g5.xlarge
INSTANCE_COUNT=1
AWS_REGION=us-east-1
SAGEMAKER_ROLE=arn:aws:iam::YOUR_ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole-XXXXX
VLLM_CONFIG_FILE=config/vllm/gemma-4-E4B.json
```

**SageMaker Role 찾기:**
```bash
# AWS CLI로 SageMaker role 찾기
aws iam list-roles | grep -i sagemaker

# 또는 AWS Console: IAM > Roles > "SageMaker" 검색
```

## 빠른 시작

### 1. Endpoint 생성 (선택사항)

SageMaker endpoint가 없다면 먼저 생성합니다:

```bash
# config/vllm/gemma-4-E4B.json과 .env 파일의 설정으로 생성
uv run python create_endpoint.py create

# 다른 vLLM 설정 파일 사용
uv run python create_endpoint.py create --vllm-config config/lmi/gpt-oss-120b.json

# 인스턴스 타입 오버라이드
uv run python create_endpoint.py create --instance-type "ml.g6.48xlarge"
```

### 2. Endpoint 테스트

Endpoint가 정상적으로 작동하는지 테스트합니다:

```bash
# .env 파일의 ENDPOINT_NAME 사용
python test_endpoint.py

# 또는 직접 지정
python test_endpoint.py --endpoint-name "your-endpoint-name"
```

### 3. 벤치마크 실행

```bash
uv run sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --dataset-name random \
  --num-prompts 50 \
  --random-input-len 2048 \
  --random-output-len 500
```

![benchmark](./imgs/sm-endpoint-bmt.png)

### 4. 상세 사용법

더 많은 예시와 파라미터 설명은 다음 문서를 참조하세요:
- **[사용 가이드 (한국어)](docs/BMT_GUIDE_ko.md)** - 상세한 사용법과 예시

## 주요 기능

- **`vllm bench serve`와 지표가 같습니다**: 공식·필드명·출력 표를 그대로 맞췄기 때문에 vLLM 실행
  결과와 나란히 놓고 비교할 수 있습니다. 같은 서버에 같은 부하를 재생해 확인했습니다
  ([vLLM 대조 검증](#vllm-대조-검증)).
- **CLI도 같습니다**: `--num-prompts`, `--request-rate`, `--burstiness`, `--max-concurrency`,
  `--percentile-metrics`, `--metric-percentiles`, `--goodput`, `--ramp-up-strategy`,
  `--ignore-eos`, `--save-result`와 샘플링 플래그가 vLLM과 동일하게 동작합니다.
- **SageMaker 전송을 제대로 다룹니다**: 블로킹 boto3 호출을 스레드 executor에서 돌려 이벤트 루프가
  요청 페이싱을 계속하고, botocore 커넥션 풀을 `--max-concurrency`에 맞춰 늘리며, 재시도를 꺼서
  조용한 재전송이 중복 집계되지 않게 합니다.
- **데이터셋**: random(`--random-input-len` / `--random-output-len` / `--random-range-ratio`),
  ShareGPT, HuggingFace.
- **엔드포인트 생애주기**: 생성, 스모크 테스트, CloudWatch 지표, Auto Scaling, MCP 서버.

## 출력 지표

정의는 vLLM과 동일합니다.

| 지표 | 정의 |
|---|---|
| **TTFT** | 내용이 있는 첫 청크 도착 − 요청 전송 |
| **TPOT** | `(latency − ttft) / (output_len − 1)`, `output_len ≤ 1`이면 가드 |
| **ITL** | 연속한 청크 사이의 간격(TTFT는 포함하지 않습니다) |
| **E2EL** | 요청 전송 → 마지막 청크 |
| **Request throughput** | 완료 요청 수 / 벤치마크 실측 시간 |
| **Output throughput** | 총 출력 토큰 / 벤치마크 실측 시간 |
| **goodput** | `--goodput ttft:…,tpot:…,e2el:…` SLO를 만족한 요청 비율 |

각 지표는 평균·중앙값·표준편차와 `--metric-percentiles`로 지정한 백분위를 함께 냅니다.

vLLM 표에 더해 **SageMaker Specifics** 절이 AWS 경계에서 생기는 것을 보고합니다. `finish_reason=length`로
잘린 요청 수, EOS로 정상 종료한 수, 컨테이너가 usage 프레임을 보내지 않은 수, 예외별 오류 분포입니다.
잘림은 오류가 아니라 결과입니다. 잘린 답변을 조용히 성공으로 세면 지연 수치가 틀어집니다.

## vLLM 대조 검증

같은 서버(L40S에서 로컬 vLLM 0.26.0으로 `google/gemma-4-E4B-it` 서빙), 같은 부하
(`--num-prompts 20 --request-rate 4 --max-concurrency 8`, random 256→128):

| | `vllm bench serve` | 이 도구 | 차이 |
|---|---|---|---|
| Total input tokens | 5307 | 5307 | 0.0% |
| Total generated tokens | 2560 | 2560 | 0.0% |
| Request throughput (req/s) | 2.76 | 2.78 | 0.6% |
| Output throughput (tok/s) | 353.78 | 355.95 | 0.6% |
| Peak concurrent requests | 12 | 12 | 0.0% |
| Mean TPOT (ms) | 16.15 | 15.88 | 1.6% |
| Median ITL (ms) | 15.85 | 15.86 | 0.0% |
| Median TTFT (ms) | 54.84 | 48.22 | 12.1% |

토큰 수가 정확히 일치하는 것이 핵심 검증입니다. 검증용 프록시가 `PayloadPart`를 **7바이트마다 일부러
쪼개기** 때문에, 각 part를 따로 파싱하는 구현이라면 여기서 토큰이 사라집니다. TTFT 차이는 대조 실행이
프록시를 한 번 더 거치기 때문입니다.

`--endpoint-url`로 SageMaker Runtime 클라이언트를 로컬 프록시로 향하게 하면 이 대조를 재현할 수 있습니다.

## `vllm bench serve`를 그냥 쓰지 않는 이유

`vllm bench serve`는 OpenAI 호환 서버에 HTTP로 말합니다. SageMaker 엔드포인트는
`boto3 invoke_endpoint_with_response_stream`으로 접근하고, 그 경계에는 고유한 함정이 있습니다.

- **`PayloadPart` 경계는 SSE 줄 경계와 맞지 않습니다.** JSON 중간에서 끊길 수 있습니다. part마다
  따로 파싱하면 토큰이 조용히 사라져 이후 모든 지표가 오염되므로, 바이트를 버퍼에 모아 `\n\n`으로
  자릅니다.
- **botocore 커넥션 풀 기본값은 10입니다.** 한 클라이언트로 64개를 동시에 보내면 54개가 클라이언트
  안에서 대기하고, 측정되는 지연은 엔드포인트가 아니라 풀의 것입니다.
- **`/invocations` 타임아웃은 60초**이고 페이로드 상한은 6 MB입니다. 한 번의 생성이 얼마나 길 수
  있는지가 여기서 정해집니다.
- **`messages` 스키마는 `max_tokens`를 씁니다.** `max_new_tokens`는 vLLM이 조용히 무시하므로 길이
  제한이 아예 적용되지 않습니다.
- **CloudWatch `ModelLatency`·`OverheadLatency`는 마이크로초입니다.** 밀리초로 다루면 1000배
  오차입니다.

## 파일 구조

```
.
├── config/                              # 설정 파일, 서빙 엔진별 폴더
│   ├── vllm/                            # 독립 vLLM DLC — SM_VLLM_* 키
│   │   ├── gemma-4-*.json.example       # 5종: E2B, E4B, 12B, 26B-A4B, 31B
│   │   └── gpt-oss-*.json.example       # 20b, 120b
│   └── lmi/                             # DJL LMI — OPTION_* 키, 같은 7개 모델
├── autoscaling/                         # Auto Scaling 관련 도구
│   ├── autoscaling.py                   # Auto Scaling 설정 스크립트
│   └── test_autoscaling.py              # Auto Scaling 테스트 스크립트
├── mcp/                                 # MCP 서버 (Model Context Protocol)
│   ├── sm_endpoint_mcp.py               # SageMaker Endpoint MCP 서버
│   └── example_mcp_config.json          # MCP 설정 예시
├── docs/                                # 문서 디렉토리
├── create_endpoint.py                   # Endpoint 생성/삭제 스크립트
├── sagemaker_benchmark.py               # 메인 벤치마크 스크립트
├── benchmark_datasets.py                # 데이터셋 모듈
├── test_endpoint.py                     # Endpoint 테스트 스크립트
├── check_metrics.py                     # CloudWatch 메트릭 체커
├── pyproject.toml                       # 프로젝트 설정 (uv)
├── requirements.txt                     # 필요한 패키지 (pip)
├── README_ko.md                         # 기본 가이드 (한국어)
├── .env.example                         # 환경 변수 예시
└── README.md                            # 기본 가이드 (영어)
```

## Endpoint 관리

### Endpoint 생성

```bash
# 기본 설정으로 생성 (config/vllm/gemma-4-E4B.json 사용)
uv run python create_endpoint.py create

# SageMaker role 지정 (로컬 환경에서 필요)
uv run python create_endpoint.py create \
  --sagemaker-role "arn:aws:iam::YOUR_ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole-XXXXX"

# 다른 vLLM 설정 파일 사용
uv run python create_endpoint.py create --vllm-config config/lmi/gpt-oss-120b.json

# 인스턴스 설정 오버라이드
uv run python create_endpoint.py create \
  --instance-type "ml.g6.48xlarge" \
  --instance-count 2

# 기본 생성 (백그라운드, 대기 안함)
uv run python create_endpoint.py create

# 완료까지 대기
uv run python create_endpoint.py create --wait true
```

### 여러 모델 설정 관리

모델 하나에 엔진별로 설정 파일 하나를 둡니다. 예시를 복사해 `.example` 를 떼고 편집합니다:

```bash
# vLLM DLC 에 gemma-4 E4B
cp config/vllm/gemma-4-E4B.json.example config/vllm/gemma-4-E4B.json

# 같은 모델을 LMI 에도 올려 컨테이너를 비교
cp config/lmi/gemma-4-E4B.json.example config/lmi/gemma-4-E4B.json

# LMI 에 gpt-oss 120B
cp config/lmi/gpt-oss-120b.json.example config/lmi/gpt-oss-120b.json

# 사용
uv run python create_endpoint.py create --vllm-config config/vllm/gemma-4-E4B.json
uv run python create_endpoint.py create --vllm-config config/lmi/gpt-oss-120b.json
```

같은 모델의 두 파일은 env 키만 다릅니다. 한쪽을 다른 쪽과 비교하면 지연에서 컨테이너가 차지하는
몫만 떼어 볼 수 있습니다.

### Endpoint 삭제

```bash
# .env 파일의 ENDPOINT_NAME 사용
uv run python create_endpoint.py delete

# 또는 직접 지정
uv run python create_endpoint.py delete --endpoint-name "your-endpoint-name"
```

### 설정 우선순위

1. CLI 인자 (최우선)
2. `.env` 파일
3. config 파일 (`--vllm-config`, 기본값 `config/vllm/gemma-4-E4B.json`)
4. 기본값

### Config 폴더 구조

서빙 컨테이너별로 폴더를 나눴습니다. 두 컨테이너는 읽는 env 키가 다르고(`SM_VLLM_*` vs `OPTION_*`),
한쪽 설정을 다른 쪽에 줘도 실패하지 않고 **기본값으로 뜹니다.** 폴더가 그 선택을 드러냅니다.

```
config/
├── vllm/                          # 독립 vLLM DLC — SM_VLLM_* 를 읽습니다
│   ├── gemma-4-E2B.json.example         effective 2.3B, 단일 GPU
│   ├── gemma-4-E4B.json.example         effective 4.5B, 단일 L4/L40S — 기본으로 권합니다
│   ├── gemma-4-12B.json.example         11.95B dense, TP 4
│   ├── gemma-4-26B-A4B.json.example     MoE, total 25.2B / active 3.8B
│   ├── gemma-4-31B.json.example         31.27B dense, L40S(44GiB) 필요
│   ├── gpt-oss-20b.json.example         MoE, TP 4
│   └── gpt-oss-120b.json.example        MoE, TP 8
└── lmi/                           # DJL LMI — OPTION_* 를 읽습니다
    ├── gemma-4-E2B.json.example         (파일 안의 LMI 버전 주의 참고)
    ├── gemma-4-E4B.json.example
    ├── gemma-4-12B.json.example
    ├── gemma-4-26B-A4B.json.example
    ├── gemma-4-31B.json.example
    ├── gpt-oss-20b.json.example
    └── gpt-oss-120b.json.example
```

두 모델군이 두 엔진에 모두 있으므로, 같은 모델을 어느 컨테이너에서 돌렸을 때 어떻게 다른지 비교할 수
있습니다.

```bash
cp config/vllm/gemma-4-E4B.json.example config/vllm/gemma-4-E4B.json
uv run python create_endpoint.py create --vllm-config config/vllm/gemma-4-E4B.json
```

각 예시에는 값의 근거를 주석으로 달았습니다: `max_num_seqs` 가 vLLM 기본값 256 이 아니라 32 인 이유,
`gpu_memory_utilization` 을 문자열로 써야 하는 이유, 31B 가 4bit 로도 44 GiB 카드를 요구하는 이유입니다.

LMI 쪽 gemma-4 파일에는 주의가 하나 붙습니다. 번들 vLLM 버전을 정하는 것은 태그의 `lmi<NN>` 부분이고,
앞의 `0.36.0` 은 djl-serving 버전입니다. gemma-4 는 vLLM >= 0.19 가 필요하니 고정한 태그의 번들 버전을
확인하세요. vLLM DLC 는 태그에 버전이 그대로 드러나서 이 계열은 `config/vllm/` 로 시작하는 편이 간단합니다.

## Auto Scaling

- 📖 **[Auto Scaling 상세 가이드 (한국어)](docs/AUTOSCALING_GUIDE_ko.md)** - 오토스케일링 설정, 테스트, 메트릭 모니터링 등 모든 기능에 대한 상세한 사용법

## MCP 서버 (Model Context Protocol)

<table>
  <tr>
    <td><img src="imgs/sm-endpoint-mcp1.png" alt="mcp1" width="450"/></td>
    <td><img src="imgs/sm-endpoint-mcp2.png" alt="mcp2" width="450"/></td>
  </tr>
  <tr>
    <td align="center"><em>LLM 호출 테스트</em></td>
    <td align="center"><em>벤치마크 예시</em></td>
  </tr>
</table>

- 📖 **[MCP 서버 가이드 (한국어)](docs/MCP_ko.md)** MCP (Model Context Protocol) 서버 활용 가이드

## LMI 아키텍처 및 vLLM 상세 

- 📖 **[LMI 아키텍처 (한국어)](docs/LMI_ARCHITECTURE_ko.md)** - 오토스케일링 설정, 테스트, 메트릭 모니터링 등 모든 기능에 대한 상세한 사용법
- 📖 **[vLLM 설정 (영어)](docs/VLLM_CONFIG.md)** - vLLM 관련 추가 설정법

## AWS 자격 증명

### 1. AWS Credentials
이 도구는 boto3를 사용하므로 AWS 자격 증명이 필요합니다:

- AWS CLI 설정 (`aws configure`)
- 환경 변수 (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- IAM 역할 (EC2/ECS에서 실행 시)

### 2. SageMaker Execution Role
Endpoint 생성 시 SageMaker execution role이 필요합니다:

**로컬 환경에서 실행:**
- `.env` 파일에 `SAGEMAKER_ROLE` 설정
- 또는 `--sagemaker-role` CLI 인자 사용

**SageMaker 환경에서 실행:**
- Notebook이나 Studio에서는 자동으로 role 감지

**Role 권한:**
SageMaker execution role에는 다음 권한이 필요합니다:
- SageMaker 리소스 생성/관리
- ECR 이미지 pull
- S3 접근 (모델 아티팩트용)

## 라이선스

본 프로젝트는 MIT 라이선스에 따라 제공됩니다. LICENSE 파일을 참조하세요.