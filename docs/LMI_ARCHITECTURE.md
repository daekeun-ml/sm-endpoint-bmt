# LMI (Large Model Inference) Architecture Guide

A detailed guide on the operational principles of LMI containers and DJL Serving architecture.

## Table of Contents

- [LMI Overview](#lmi-overview)
- [DJL Serving Architecture](#djl-serving-architecture)
- [LMI Components](#lmi-components)
- [Engine Operation Modes](#engine-operation-modes)
- [Backend Selection Guide](#backend-selection-guide)
- [Workflow](#workflow)

---

## LMI Overview

LMI (Large Model Inference) containers are high-performance Docker containers designed to efficiently deploy and run large language models (LLMs) on AWS SageMaker.

### Key Features

- **Unified Solution**: Integrates Model Server + Inference Library + Handler into one
- **Multiple Backend Support**: vLLM, TensorRT-LLM, Transformers NeuronX
- **Optimized Performance**: Continuous Batching, Token Streaming, Quantization
- **Multi-GPU Support**: Distributed processing of large models through Tensor Parallelism
- **Unified Configuration Format**: Use various backends with the same configuration approach

---

## DJL Serving Architecture

LMI containers use [DJL Serving](https://github.com/deepjavalibrary/djl-serving) as the Model Server.

### Overall Architecture

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

### 4-Layer Structure

#### 1. Frontend Layer
- **Netty HTTP Server**: Receives and manages incoming requests
- **Request Handlers**:
  - Inference API: Handles inference requests
  - Management API: Endpoint and workflow CRUD operations
  - Plugin APIs: Extension functionality

#### 2. Workflows Layer
- Execution pipeline combining multiple models and code
- Configured via `workflow.json` file
- Same model can be shared across multiple workflows

#### 3. WorkLoadManager (WLM) Layer
- **Worker Pool**: Thread pool per model
- **Worker Group**: Worker groups per device (CPU, GPU, etc.)
- **Worker Thread**: Threads that perform actual inference
- **Auto-scaling**: Automatic adjustment of min/max worker count
- **Batching & Routing**: Request batch processing and routing

#### 4. Engine Layer
- **DJL Predictor**: Supports various frameworks
- **Python Engine**: Executes Python-based inference libraries
- **MPI Engine**: MPI mode for distributed inference

---

## LMI Components

### 1. Model Server (DJL Serving)

DJL Serving manages request routing with a Netty-based frontend, while backend workers execute model inference.

**Key Functions:**
- Request routing to backend workers
- Model worker scaling
- Job queue management
- Health check and monitoring

### 2. Python Engine & Inference Backends

Utilizes Python-based inference libraries through Python Engine.

**Supported Backends:**
- **vLLM**: GPU optimized, Ray-based multiprocess
- **TensorRT-LLM**: NVIDIA GPU optimized, MPI-based
- **Transformers NeuronX**: AWS Inferentia optimized

**Backend Definition:**
```
Backend = Engine (Python/MPI) + Inference Library (vLLM/TRT-LLM/TNX)
```

### 3. Built-In Handlers

LMI provides built-in inference handlers for each backend.

**Handler Responsibilities:**
- Configuration parsing
- Model loading and optimization
- Inference execution
- Response formatting

**Advantages:**
- No need to learn each library individually
- Easy backend switching with unified configuration format
- Consistent experience when integrating new libraries

### 4. Configuration

Configuration defines the entire deployment environment.

**Configuration Methods:**
1. `serving.properties` file
2. Environment Variables

**Configuration Items:**
- Model artifact location (HuggingFace Model ID, S3 URI)
- Model Server settings (queue size, auto-scaling)
- Engine/Backend settings (quantization, tensor parallel degree, etc.)

---

## Engine Operation Modes

LMI provides two engine operation modes.

### Python Engine Mode

**Features:**
- Independent Python process execution
- GPU allocation via `CUDA_VISIBLE_DEVICES`
- Communication through socket connections

**Architecture:**
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

**Supported Backends:**
- vLLM (uses Ray internally)
- HuggingFace Accelerate
- Transformers NeuronX

**Configuration:**
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

**Default Behavior:**
Automatically uses Python mode when `option.model_id` is specified

### MPI Engine Mode

**Features:**
- MPI (Multi-Process-Interface/Inference)
- Multiple processes executed via `mpirun`
- Process count = `tensor_parallel_degree`
- Socket connection to each process
- Results received only from Rank 0

**Architecture:**
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

**Supported Backends:**
- TensorRT-LLM (MPI-based distributed inference)

**Configuration:**
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

**Auto-scaling Support:**
Can configure multiple workers to use different GPUs

---

## Backend Selection Guide

### vLLM (Python Engine)

**Recommended Use Cases:**
- GPU-based inference
- High throughput requirements
- Continuous batching needed
- PagedAttention optimization utilization

**Features:**
- Ray-based multiprocess
- Automatic memory management
- Prefix caching support
- FP8, AWQ, GPTQ quantization

**Container:**
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.36.0-lmi26.0.0-cu130
```

### TensorRT-LLM (MPI Engine)

**Recommended Use Cases:**
- NVIDIA GPU optimization
- Lowest latency requirements
- Large models (70B+)
- Ahead-of-time compilation

**Features:**
- MPI-based distributed inference
- TensorRT optimization
- FP8, INT8 quantization
- Speculative decoding

**Container:**
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.33.0-tensorrtllm0.21.0-cu128
```

### Transformers NeuronX (Python Engine)

**Recommended Use Cases:**
- AWS Inferentia/Trainium usage
- Cost optimization
- Large-scale deployment

**Features:**
- Neuron core optimization
- Multithreading-based
- Low cost
- AWS native integration

**Container:**
```
763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.30.0-neuronx-sdk2.20.1
```

---

## Workflow

### Request Processing Flow

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

### Batch Processing (Batching)

```
Request 1 ─┐
Request 2 ─┤
Request 3 ─┼─→ Job Queue ─→ Batch ─→ Worker Thread ─→ Inference
Request 4 ─┤
Request 5 ─┘
```

**Continuous Batching:**
- Add new requests immediately when requests complete
- Minimize waiting time
- Maximize throughput

### Auto-scaling

```
Low Load:  [Worker 1] [Worker 2]
           (Min Workers)

High Load: [Worker 1] [Worker 2] [Worker 3] [Worker 4]
           (Scaled Up to Max Workers)

Load Drop: [Worker 1] [Worker 2]
           (Scaled Down)
```

**Configuration:**
- `min_workers`: Minimum number of workers
- `max_workers`: Maximum number of workers
- Automatically adjusts based on load

---

## Simple Deployment Example

### Minimal Configuration (Using HF_MODEL_ID only)

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

### Advanced Configuration

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

## References

- [DJL Serving Architecture](https://docs.djl.ai/master/docs/serving/serving/docs/architecture.html)
- [LMI Deployment Guide](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/deployment_guide/index.html)
- [LMI Engine Guide](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/conceptual_guide/lmi_engine.html)
- [LMI Starting Guide](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/starting-guide.html)
- [DJL Serving GitHub](https://github.com/deepjavalibrary/djl-serving)