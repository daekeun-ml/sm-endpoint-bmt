# LMI (Large Model Inference) 아키텍처 가이드

LMI 컨테이너의 동작 원리와 DJL Serving 아키텍처에 대한 상세 가이드입니다.

## 목차

- [LMI 개요](#lmi-개요)
- [DJL Serving 아키텍처](#djl-serving-아키텍처)
- [LMI 컴포넌트](#lmi-컴포넌트)
- [Engine 동작 모드](#engine-동작-모드)
- [Backend 선택 가이드](#backend-선택-가이드)
- [워크플로우](#워크플로우)

---

## LMI 개요

LMI (Large Model Inference) 컨테이너는 대규모 언어 모델(LLM)을 AWS SageMaker에서 효율적으로 배포하고 실행하기 위한 고성능 Docker 컨테이너입니다.

### 주요 특징

- **통합 솔루션**: Model Server + Inference Library + Handler를 하나로 통합
- **다양한 Backend 지원**: vLLM, TensorRT-LLM, Transformers NeuronX
- **최적화된 성능**: Continuous Batching, Token Streaming, Quantization
- **Multi-GPU 지원**: Tensor Parallelism을 통한 대규모 모델 분산 처리
- **통합 설정 형식**: 다양한 Backend를 동일한 설정 방식으로 사용

---

## DJL Serving 아키텍처

LMI 컨테이너는 [DJL Serving](https://github.com/deepjavalibrary/djl-serving)을 Model Server로 사용합니다.

### 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        DJL Serving                          │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Netty HTTP Server)                               │
│  ├─ Inference API Handler                                   │
│  ├─ Management API Handler                                  │
│  └─ Plugin Handlers                                         │
├─────────────────────────────────────────────────────────────┤
│  Workflows                                                   │
│  └─ Multiple Models + Glue Code Pipeline                    │
├─────────────────────────────────────────────────────────────┤
│  WorkLoadManager (WLM)                                       │
│  ├─ Worker Pool (per model)                                 │
│  │  ├─ Job Queue                                            │
│  │  └─ Worker Groups (per device)                           │
│  │     └─ Worker Threads                                    │
│  │        └─ DJL Predictor                                  │
│  │           └─ Python Engine / MPI Engine                  │
│  │              └─ Inference Library (vLLM, TRT-LLM, etc)   │
└─────────────────────────────────────────────────────────────┘
```

### 4개 레이어 구조

#### 1. Frontend Layer
- **Netty HTTP Server**: 들어오는 요청을 받아 관리
- **Request Handlers**:
  - Inference API: 추론 요청 처리
  - Management API: 엔드포인트 및 워크플로우 CRUD
  - Plugin APIs: 확장 기능

#### 2. Workflows Layer
- 여러 모델과 코드를 조합한 실행 파이프라인
- `workflow.json` 파일로 설정
- 동일 모델을 여러 워크플로우에서 공유 가능

#### 3. WorkLoadManager (WLM) Layer
- **Worker Pool**: 모델별 워커 스레드 풀
- **Worker Group**: 디바이스별 워커 그룹 (CPU, GPU 등)
- **Worker Thread**: 실제 추론을 수행하는 스레드
- **Auto-scaling**: 최소/최대 워커 수 자동 조정
- **Batching & Routing**: 요청 배치 처리 및 라우팅

#### 4. Engine Layer
- **DJL Predictor**: 다양한 프레임워크 지원
- **Python Engine**: Python 기반 추론 라이브러리 실행
- **MPI Engine**: 분산 추론을 위한 MPI 모드

---

## LMI 컴포넌트

### 1. Model Server (DJL Serving)

DJL Serving은 Netty 기반 프론트엔드로 요청 라우팅을 관리하고, 백엔드 워커가 모델 추론을 실행합니다.

**주요 기능:**
- Request routing to backend workers
- Model worker scaling
- Job queue management
- Health check and monitoring

### 2. Python Engine & Inference Backends

Python Engine을 통해 Python 기반 추론 라이브러리를 활용합니다.

**지원 Backend:**
- **vLLM**: GPU 최적화, Ray 기반 멀티프로세스
- **TensorRT-LLM**: NVIDIA GPU 최적화, MPI 기반
- **Transformers NeuronX**: AWS Inferentia 최적화

**Backend 정의:**
```
Backend = Engine (Python/MPI) + Inference Library (vLLM/TRT-LLM/TNX)
```

### 3. Built-In Handlers

LMI는 각 Backend에 대한 내장 추론 핸들러를 제공합니다.

**핸들러 역할:**
- 설정 파싱
- 모델 로딩 및 최적화
- 추론 실행
- 응답 포맷팅

**장점:**
- 각 라이브러리를 개별적으로 학습할 필요 없음
- 통합된 설정 형식으로 쉽게 Backend 전환 가능
- 새로운 라이브러리 통합 시 일관된 경험 제공

### 4. Configuration

설정은 전체 배포 환경을 정의합니다.

**설정 방법:**
1. `serving.properties` 파일
2. 환경 변수 (Environment Variables)

**설정 항목:**
- 모델 아티팩트 위치 (HuggingFace Model ID, S3 URI)
- Model Server 설정 (queue size, auto-scaling)
- Engine/Backend 설정 (quantization, tensor parallel degree 등)

---

## Engine 동작 모드

LMI는 두 가지 Engine 동작 모드를 제공합니다.

### Python Engine Mode

**특징:**
- 독립적인 Python 프로세스 실행
- `CUDA_VISIBLE_DEVICES`로 GPU 할당
- Socket 연결로 통신

**아키텍처:**
```
┌─────────────────────────────────────────┐
│         DJL Serving (Java)              │
│  ┌───────────────────────────────────┐  │
│  │   WorkLoadManager                 │  │
│  │  ┌─────────────────────────────┐  │  │
│  │  │  Worker Thread 1            │  │  │
│  │  │  └─ Socket Connection       │  │  │
│  │  └─────────────────────────────┘  │  │
│  │  ┌─────────────────────────────┐  │  │
│  │  │  Worker Thread 2            │  │  │
│  │  │  └─ Socket Connection       │  │  │
│  │  └─────────────────────────────┘  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
           │                    │
           ▼                    ▼
    ┌──────────────┐    ┌──────────────┐
    │ Python       │    │ Python       │
    │ Process 1    │    │ Process 2    │
    │ GPU 0        │    │ GPU 1        │
    │ (vLLM/Ray)   │    │ (vLLM/Ray)   │
    └──────────────┘    └──────────────┘
```

**사용 Backend:**
- vLLM (내부적으로 Ray 사용)
- HuggingFace Accelerate
- Transformers NeuronX

**설정:**
```properties
# serving.properties
engine=Python
option.mpi_mode=false
```

```bash
# Environment Variables
OPTION_ENGINE=Python
OPTION_MPI_MODE=false
```

**기본 동작:**
`option.model_id`를 지정하면 자동으로 Python 모드 사용

### MPI Engine Mode

**특징:**
- MPI (Multi-Process-Interface/Inference)
- `mpirun`으로 여러 프로세스 실행
- 프로세스 수 = `tensor_parallel_degree`
- 각 프로세스에 Socket 연결
- Rank 0에서만 결과 수신

**아키텍처:**
```
┌─────────────────────────────────────────┐
│         DJL Serving (Java)              │
│  ┌───────────────────────────────────┐  │
│  │   WorkLoadManager                 │  │
│  │  ┌─────────────────────────────┐  │  │
│  │  │  Worker Thread              │  │  │
│  │  │  └─ Multiple Sockets        │  │  │
│  │  └─────────────────────────────┘  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
           │         │         │
           ▼         ▼         ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ MPI      │ │ MPI      │ │ MPI      │
    │ Rank 0   │ │ Rank 1   │ │ Rank 2   │
    │ GPU 0    │ │ GPU 1    │ │ GPU 2    │
    │(TRT-LLM) │ │(TRT-LLM) │ │(TRT-LLM) │
    └──────────┘ └──────────┘ └──────────┘
         └──────────┬──────────┘
              Tensor Parallel
```

**사용 Backend:**
- TensorRT-LLM (MPI 기반 분산 추론)

**설정:**
```properties
# serving.properties
engine=Python
option.mpi_mode=true
```

```bash
# Environment Variables
OPTION_ENGINE=Python
OPTION_MPI_MODE=true
```

**Auto-scaling 지원:**
여러 워커가 서로 다른 GPU를 사용하도록 설정 가능

---

## Backend 선택 가이드

### vLLM (Python Engine)

**추천 사용 사례:**
- GPU 기반 추론
- 높은 처리량 요구
- Continuous batching 필요
- PagedAttention 최적화 활용

**특징:**
- Ray 기반 멀티프로세스
- 자동 메모리 관리
- Prefix caching 지원
- FP8, AWQ, GPTQ 양자화

**컨테이너:**
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.36.0-lmi26.0.0-cu130
```

### TensorRT-LLM (MPI Engine)

**추천 사용 사례:**
- NVIDIA GPU 최적화
- 최저 지연시간 요구
- 대규모 모델 (70B+)
- Ahead-of-time compilation

**특징:**
- MPI 기반 분산 추론
- TensorRT 최적화
- FP8, INT8 양자화
- Speculative decoding

**컨테이너:**
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.33.0-tensorrtllm0.21.0-cu128
```

### Transformers NeuronX (Python Engine)

**추천 사용 사례:**
- AWS Inferentia/Trainium 사용
- 비용 최적화
- 대규모 배포

**특징:**
- Neuron 코어 최적화
- 멀티스레딩 기반
- 낮은 비용
- AWS 네이티브 통합

**컨테이너:**
```
763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.30.0-neuronx-sdk2.20.1
```

---

## 워크플로우

### 요청 처리 흐름

```
1. Client Request
   ↓
2. Netty Frontend (HTTP Server)
   ↓
3. Inference API Handler
   ↓
4. Workflow Selection (Round-robin)
   ↓
5. WorkLoadManager
   ├─ Job Queue
   └─ Worker Pool
      ├─ Worker Group (Device)
      └─ Worker Thread
         └─ DJL Predictor
            └─ Python/MPI Engine
               └─ Inference Library (vLLM/TRT-LLM/TNX)
                  └─ Model Inference
   ↓
6. Response to Client
```

### 배치 처리 (Batching)

```
Request 1 ─┐
Request 2 ─┤
Request 3 ─┼─→ Job Queue ─→ Batch ─→ Worker Thread ─→ Inference
Request 4 ─┤
Request 5 ─┘
```

**Continuous Batching:**
- 요청이 완료되면 즉시 새 요청 추가
- 대기 시간 최소화
- 처리량 최대화

### Auto-scaling

```
Low Load:  [Worker 1] [Worker 2]
           (Min Workers)

High Load: [Worker 1] [Worker 2] [Worker 3] [Worker 4]
           (Scaled Up to Max Workers)

Load Drop: [Worker 1] [Worker 2]
           (Scaled Down)
```

**설정:**
- `min_workers`: 최소 워커 수
- `max_workers`: 최대 워커 수
- 부하에 따라 자동 조정

---

## 간단한 배포 예시

### 최소 설정 (HF_MODEL_ID만 사용)

```python
from sagemaker.djl_inference import DJLModel

model = DJLModel(
    model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
    role=iam_role,
    env={
        "HF_TOKEN": "<your-token>",
    }
)

predictor = model.deploy(
    instance_type="ml.g5.12xlarge",
    initial_instance_count=1
)
```

### 고급 설정

```python
model = DJLModel(
    model_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
    role=iam_role,
    env={
        "HF_TOKEN": "<your-token>",
        "TENSOR_PARALLEL_DEGREE": "8",
        "MAX_BATCH_SIZE": "256",
        "MAX_CONCURRENT_REQUESTS": "1000",
    }
)
```

---

## 참고 자료

- [DJL Serving Architecture](https://docs.djl.ai/master/docs/serving/serving/docs/architecture.html)
- [LMI Deployment Guide](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/deployment_guide/index.html)
- [LMI Engine Guide](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/conceptual_guide/lmi_engine.html)
- [LMI Starting Guide](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/starting-guide.html)
- [DJL Serving GitHub](https://github.com/deepjavalibrary/djl-serving)
