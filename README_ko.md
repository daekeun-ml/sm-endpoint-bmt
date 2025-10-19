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

### 1. vLLM 설정 파일

`config/` 폴더에서 예시 파일을 복사하여 설정 파일을 만듭니다:

```bash
# 20B 모델 설정
cp config/vllm_config.json.example config/vllm_config.json

# 또는 120B 모델 설정
cp config/vllm_config_120b.json.example config/vllm_config_120b.json
```

`config/vllm_config.json` 파일 예시:
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
VLLM_CONFIG_FILE=config/vllm_config.json
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
# config/vllm_config.json과 .env 파일의 설정으로 생성
uv run python create_endpoint.py create

# 다른 vLLM 설정 파일 사용
uv run python create_endpoint.py create --vllm-config config/vllm_config_120b.json

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
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --num-prompts 200
```

### 4. 고급 사용법

더 많은 예시와 파라미터 설명은 다음 문서를 참조하세요:
- **[사용 가이드 (한국어)](docs/USAGE_KR.md)** - 상세한 사용법과 예시

## 주요 기능

- ✅ **vLLM 호환**: vLLM bench serve와 동일한 CLI 인터페이스
- ✅ **Auto Scaling**: CloudWatch 메트릭 기반 자동 스케일링
- ✅ **MCP 서버**: Kiro IDE와 통합된 Model Context Protocol 서버
- ✅ **다양한 데이터셋**: Random, ShareGPT, HuggingFace 데이터셋 지원
- ✅ **정확한 메트릭**: TTFT, TPOT, ITL 등 상세한 성능 메트릭
- ✅ **샘플링 파라미터**: Temperature, top-p, top-k, beam search 등
- ✅ **동시성 제어**: 최대 동시 요청 수 및 요청 속도 제어

## 출력 메트릭

- **Request throughput**: 초당 요청 처리량
- **Token throughput**: 초당 토큰 처리량
- **TTFT (Time to First Token)**: 첫 토큰까지의 시간
- **TPOT (Time per Output Token)**: 출력 토큰당 시간
- **ITL (Inter-token Latency)**: 토큰 간 지연 시간

각 메트릭은 평균, 중앙값, P99 값을 제공합니다.

## 파일 구조

```
.
├── config/                              # 설정 파일 디렉토리
│   ├── vllm_config.json.example         # vLLM 설정 예시 (기본)
│   ├── vllm_config_120b.json.example    # vLLM 설정 예시 (120B)
│   └── vllm_config.json                 # 실제 vLLM 설정
├── autoscaling/                         # Auto Scaling 관련 도구
│   ├── autoscaling.py                   # Auto Scaling 설정 스크립트
│   ├── test_autoscaling.py              # Auto Scaling 테스트 스크립트
│   └── README.md                        # Auto Scaling 사용 가이드
├── mcp/                                 # MCP 서버 (Model Context Protocol)
│   ├── sm_endpoint_mcp.py               # SageMaker Endpoint MCP 서버
│   ├── example_mcp_config.json          # MCP 설정 예시
│   └── README.md                        # MCP 서버 사용 가이드
├── docs/                                # 문서 디렉토리
│   ├── USAGE.md                         # 상세 사용 가이드
│   ├── AUTOSCALING_GUIDE.md             # Auto Scaling 가이드
│   ├── VLLM_CONFIG.md                   # vLLM 설정 가이드 (LMI v16)
│   └── LMI_ARCHITECTURE.md              # LMI 아키텍처 가이드
├── create_endpoint.py                   # Endpoint 생성/삭제 스크립트
├── sagemaker_benchmark.py               # 메인 벤치마크 스크립트
├── benchmark_datasets.py                # 데이터셋 모듈
├── test_endpoint.py                     # Endpoint 테스트 스크립트
├── check_metrics.py                     # CloudWatch 메트릭 체커
├── create_endpoint_tutorial.ipynb       # Endpoint 생성 튜토리얼
├── pyproject.toml                       # 프로젝트 설정 (uv)
├── requirements.txt                     # 필요한 패키지 (pip)
├── .env.example                         # 환경 변수 예시
└── README.md                            # 기본 가이드
```

## Endpoint 관리

### Endpoint 생성

```bash
# 기본 설정으로 생성 (config/vllm_config.json 사용)
uv run python create_endpoint.py create

# SageMaker role 지정 (로컬 환경에서 필요)
uv run python create_endpoint.py create \
  --sagemaker-role "arn:aws:iam::YOUR_ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole-XXXXX"

# 다른 vLLM 설정 파일 사용
uv run python create_endpoint.py create --vllm-config config/vllm_config_120b.json

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

`config/` 폴더에서 다양한 모델을 위한 설정 파일을 관리할 수 있습니다:

```bash
# 20B 모델
cp config/vllm_config.json.example config/vllm_config_20b.json
nano config/vllm_config_20b.json

# 120B 모델
cp config/vllm_config_120b.json.example config/vllm_config_120b.json
nano config/vllm_config_120b.json

# 사용
uv run python create_endpoint.py create --vllm-config config/vllm_config_20b.json
uv run python create_endpoint.py create --vllm-config config/vllm_config_120b.json
```

### Endpoint 삭제

```bash
# .env 파일의 ENDPOINT_NAME 사용
uv run python create_endpoint.py delete

# 또는 직접 지정
uv run python create_endpoint.py delete --endpoint-name "your-endpoint-name"
```

### 상세 사용법

- 📖 **[SageMaker Endpoint 벤치마크 도구 - 사용 가이드 (한국어)](docs/BMT_GUIDE_ko.md)**

### 설정 우선순위

1. CLI 인자 (최우선)
2. `.env` 파일
3. `config/vllm_config.json` 파일
4. 기본값

### Config 폴더 구조

`config/` 폴더에 여러 모델 설정을 관리할 수 있습니다:

```
config/
├── vllm_config.json              # 기본 설정
├── vllm_config_20b.json          # 20B 모델 설정
├── vllm_config_120b.json         # 120B 모델 설정
└── vllm_config_custom.json       # 커스텀 설정
```

## Auto Scaling

- 📖 **[Auto Scaling 상세 가이드 (한국어)](docs/AUTOSCALING_GUIDE_ko.md)** - 오토스케일링 설정, 테스트, 메트릭 모니터링 등 모든 기능에 대한 상세한 사용법

## MCP 서버 (Model Context Protocol)

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

MIT

## 참고

이 프로젝트는 [vLLM](https://github.com/vllm-project/vllm)의 벤치마크 도구를 참고하여 만들어졌습니다.
