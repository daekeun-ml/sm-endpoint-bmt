# SageMaker Endpoint MCP Server

이 MCP 서버는 SageMaker 엔드포인트의 메트릭 확인과 벤치마킹 기능을 MCP(Model Context Protocol)를 통해 제공합니다.

## 기능

### 🔍 메트릭 확인 도구
- `check_endpoint_metrics`: CloudWatch 메트릭 수집 및 분석
- `generate_metrics_report`: 상세한 마크다운 리포트 생성
- `get_endpoint_status`: 엔드포인트 상태 정보 조회

### 🚀 벤치마킹 도구
- `run_benchmark`: 성능 벤치마크 실행
- `list_available_datasets`: 사용 가능한 데이터셋 목록

## 설치 및 설정

### 1. 의존성 설치

```bash
# FastMCP 설치
pip install fastmcp

# 또는 uv 사용 (권장)
uv add fastmcp
```

### 2. MCP 서버 설정

#### 워크스페이스 레벨 설정 (.kiro/settings/mcp.json)

```json
{
  "mcpServers": {
    "sm-endpoint-tools": {
      "command": "python",
      "args": ["mcp/sm_endpoint_mcp.py"],
      "cwd": ".",
      "env": {
        "AWS_REGION": "us-east-1"
      },
      "disabled": false,
      "autoApprove": [
        "check_endpoint_metrics",
        "get_endpoint_status",
        "list_available_datasets"
      ]
    }
  }
}
```

#### 사용자 레벨 설정 (~/.kiro/settings/mcp.json)

```json
{
  "mcpServers": {
    "sm-endpoint-tools": {
      "command": "uvx",
      "args": ["--from", "fastmcp", "python", "/path/to/sm-endpoint-bmt/mcp/sm_endpoint_mcp.py"],
      "env": {
        "AWS_REGION": "us-east-1",
        "AWS_PROFILE": "default"
      },
      "disabled": false,
      "autoApprove": [
        "check_endpoint_metrics",
        "get_endpoint_status"
      ]
    }
  }
}
```

### 3. 환경 변수 설정

MCP 서버가 AWS에 접근할 수 있도록 환경 변수를 설정하세요:

```bash
# AWS 자격 증명 (다음 중 하나)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
# 또는
export AWS_PROFILE="your-profile"

# AWS 리전
export AWS_REGION="us-east-1"
```

## 사용법

### 메트릭 확인

```
엔드포인트 "my-endpoint"의 최근 30분 메트릭을 확인해줘
```

```
"gpt-model-endpoint"의 한국어 메트릭 리포트를 생성해줘
```

```
cloudwatch로 최근 2분간의 엔드포인트 호출 내역 찍어봐
```

**참고:** CloudWatch 메트릭은 실시간이 아니며 5-15분의 지연이 있을 수 있습니다.

### 벤치마크 실행

```
엔드포인트 "my-endpoint"에서 100개 요청으로 벤치마크를 실행해줘
```

```
"llm-endpoint"에서 동시 요청 5개로 성능 테스트를 해줘
```

```
huggingface 데이터셋 tatsu-lab/alpaca 으로 50건 벤치마크도 해봐
```

**실제 테스트 결과 예시 (120B 모델):**
- 50개 요청, 동시 사용자 2명
- 평균 응답시간: ~1.3초
- 처리량: 0.75 RPS
- 성공률: 100%

### 엔드포인트 테스트

```
sagemaker endpoint gpt-oss-120b-2025-10-16-10-23-39-438 에 대해 간단한 호출 테스트해줘
```

### 엔드포인트 상태 확인

```
"my-endpoint"의 현재 상태를 확인해줘
```

## 도구 상세 설명

### test_endpoint

SageMaker 엔드포인트에 간단한 테스트 요청을 보내 정상 작동을 확인합니다.

**매개변수:**
- `endpoint_name` (필수): SageMaker 엔드포인트 이름
- `test_payload` (선택): 테스트용 JSON 페이로드
- `region` (선택): AWS 리전

**반환값:**
- 응답 시간
- 생성된 텍스트
- 성공/실패 상태

### check_endpoint_metrics

CloudWatch에서 SageMaker 엔드포인트 메트릭을 수집하고 분석합니다.

**매개변수:**
- `endpoint_name` (필수): SageMaker 엔드포인트 이름
- `minutes` (선택, 기본값: 30): 확인할 시간 범위 (분)
- `region` (선택): AWS 리전
- `language` (선택, 기본값: "en"): 언어 ("en" 또는 "ko")

**반환값:**
- 메트릭 요약 정보
- 상세 메트릭 데이터
- 엔드포인트 상태 정보

### generate_metrics_report

상세한 마크다운 형식의 메트릭 리포트를 생성합니다.

**매개변수:**
- `endpoint_name` (필수): SageMaker 엔드포인트 이름
- `minutes` (선택, 기본값: 30): 확인할 시간 범위 (분)
- `region` (선택): AWS 리전
- `language` (선택, 기본값: "en"): 언어 ("en" 또는 "ko")

**반환값:**
- 마크다운 형식의 상세 리포트
- 메타데이터

### run_benchmark

SageMaker 엔드포인트에서 성능 벤치마크를 실행합니다.

**매개변수:**
- `endpoint_name` (필수): SageMaker 엔드포인트 이름
- `num_requests` (선택, 기본값: 10): 전송할 요청 수
- `concurrent_requests` (선택, 기본값: 1): 동시 요청 수
- `region` (선택): AWS 리전
- `dataset_name` (선택): HuggingFace 데이터셋 이름 (예: 'tatsu-lab/alpaca')

**반환값:**
- 벤치마크 요약 결과
- 상세 성능 메트릭
- 지연 시간 통계

### get_endpoint_status

SageMaker 엔드포인트의 현재 상태를 조회합니다.

**매개변수:**
- `endpoint_name` (필수): SageMaker 엔드포인트 이름
- `region` (선택): AWS 리전

**반환값:**
- 엔드포인트 상태 정보
- 인스턴스 정보
- 설정 정보

### list_available_datasets

벤치마킹에 사용 가능한 데이터셋 목록을 반환합니다.

**반환값:**
- 내장 데이터셋 목록
- HuggingFace 데이터셋 사용법
- 각 데이터셋의 설명

## 문제 해결

### 1. MCP 서버 연결 실패

**증상:** MCP 서버에 연결할 수 없음

**해결 방법:**
1. 경로 확인: `mcp/sm_endpoint_mcp.py` 파일이 존재하는지 확인
2. 권한 확인: 파일 실행 권한이 있는지 확인
3. 의존성 확인: FastMCP와 필요한 패키지가 설치되어 있는지 확인

```bash
# 파일 권한 설정
chmod +x mcp/sm_endpoint_mcp.py

# 의존성 확인
python -c "import fastmcp; print('FastMCP OK')"
```

### 2. AWS 자격 증명 오류

**증상:** AWS 접근 권한 오류

**해결 방법:**
1. AWS 자격 증명 확인:
   ```bash
   aws sts get-caller-identity
   ```

2. 환경 변수 확인:
   ```bash
   echo $AWS_REGION
   echo $AWS_PROFILE
   ```

3. IAM 권한 확인:
   - SageMaker 읽기 권한
   - CloudWatch 메트릭 읽기 권한

### 3. 모듈 import 오류

**증상:** `check_metrics` 또는 `sagemaker_benchmark` 모듈을 찾을 수 없음

**해결 방법:**
1. 작업 디렉토리 확인: MCP 서버가 프로젝트 루트에서 실행되는지 확인
2. 경로 설정: `cwd` 설정이 올바른지 확인
3. 파일 존재 확인: 필요한 Python 파일들이 존재하는지 확인

### 4. 성능 최적화

**느린 응답 시간:**
1. 메트릭 조회 시간 범위 줄이기 (`minutes` 매개변수)
2. 벤치마크 요청 수 줄이기 (`num_prompts` 매개변수)
3. 동시 요청 수 조정 (`concurrent_requests` 매개변수)

## 보안 고려사항

1. **자격 증명 관리**: AWS 자격 증명을 안전하게 관리하세요
2. **권한 최소화**: 필요한 최소 권한만 부여하세요
3. **자동 승인**: 신뢰할 수 있는 도구만 `autoApprove`에 추가하세요

## 개발 및 확장

### 새로운 도구 추가

```python
@mcp.tool()
def your_new_tool(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Your tool description
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
    
    Returns:
        Dictionary containing results
    """
    try:
        # Your implementation here
        result = do_something(param1, param2)
        
        return {
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
```

### 로깅 추가

```python
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 도구에서 로깅 사용
logger.info(f"Running benchmark for endpoint: {endpoint_name}")
```

## 참고 자료

- [FastMCP 문서](https://github.com/jlowin/fastmcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [SageMaker 개발자 가이드](https://docs.aws.amazon.com/sagemaker/)
- [CloudWatch 메트릭](https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html)