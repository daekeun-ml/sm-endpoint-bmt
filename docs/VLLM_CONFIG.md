# vLLM Configuration Guide

Complete guide for vLLM environment variables in LMI (Large Model Inference) containers.

Based on:
- [DJL LMI vLLM User Guide](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/vllm_user_guide.html)
- vLLM 0.10.2 (LMI v16)

## Table of Contents

- [Configuration Examples](#configuration-examples)
- [Required Configuration](#required-configuration)
- [Core Configuration](#core-configuration)
- [Performance Tuning](#performance-tuning)
- [Quantization](#quantization)
- [LoRA Adapters](#lora-adapters)
- [Advanced Configuration](#advanced-configuration)
- [vLLM Engine Arguments](#vllm-engine-arguments)
- [Troubleshooting](#troubleshooting)

---

## Configuration Examples

### Example 1: Basic Async Mode (Recommended)
```json
{
  "HF_MODEL_ID": "openai/gpt-oss-20b",
  "HF_TOKEN": "",
  "OPTION_ASYNC_MODE": "true",
  "OPTION_ROLLING_BATCH": "disable",
  "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
  "TENSOR_PARALLEL_DEGREE": "max",
  "OPTION_MODEL_LOADING_TIMEOUT": "1500",
  "SERVING_FAIL_FAST": "true"
}
```

### Example 2: High Performance Configuration
```json
{
  "HF_MODEL_ID": "meta-llama/Llama-2-70b-hf",
  "OPTION_ASYNC_MODE": "true",
  "OPTION_ROLLING_BATCH": "disable",
  "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
  "TENSOR_PARALLEL_DEGREE": "8",
  "GPU_MEMORY_UTILIZATION": "0.95",
  "OPTION_MAX_MODEL_LEN": "4096",
  "OPTION_MAX_NUM_BATCHED_TOKENS": "32768",
  "OPTION_ENABLE_PREFIX_CACHING": "true",
  "OPTION_ENABLE_CHUNKED_PREFILL": "true"
}
```

### Example 3: Quantized Model (FP8)
```json
{
  "HF_MODEL_ID": "neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8",
  "OPTION_ASYNC_MODE": "true",
  "OPTION_ROLLING_BATCH": "disable",
  "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
  "TENSOR_PARALLEL_DEGREE": "max",
  "GPU_MEMORY_UTILIZATION": "0.9"
}
```
**Note:** No `OPTION_QUANTIZE` set - vLLM auto-detects and uses Marlin kernels

### Example 4: LoRA Adapters
```json
{
  "HF_MODEL_ID": "meta-llama/Llama-2-7b-hf",
  "OPTION_ASYNC_MODE": "true",
  "OPTION_ROLLING_BATCH": "disable",
  "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
  "TENSOR_PARALLEL_DEGREE": "1",
  "OPTION_ENABLE_LORA": "true",
  "OPTION_MAX_LORAS": "4",
  "OPTION_MAX_LORA_RANK": "64",
  "GPU_MEMORY_UTILIZATION": "0.85"
}
```

### Example 5: Large Model with CPU Offload
```json
{
  "HF_MODEL_ID": "openai/gpt-oss-120b",
  "OPTION_ASYNC_MODE": "true",
  "OPTION_ROLLING_BATCH": "disable",
  "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",
  "TENSOR_PARALLEL_DEGREE": "8",
  "GPU_MEMORY_UTILIZATION": "0.9",
  "OPTION_CPU_OFFLOAD_GB_PER_GPU": "4",
  "OPTION_MODEL_LOADING_TIMEOUT": "2000"
}
```

---

## Required Configuration

### HF_MODEL_ID
**Type:** String  
**Required:** Yes  
**Example:** `"openai/gpt-oss-20b"`, `"meta-llama/Llama-2-7b-hf"`  
**Description:** HuggingFace model ID or path to model artifacts

### OPTION_ENTRYPOINT
**Type:** String  
**Required:** Yes (for async mode)  
**Default:** `"djl_python.lmi_vllm.vllm_async_service"`  
**Description:** Entry point for vLLM async service (recommended mode)

---

## Core Configuration

### OPTION_ASYNC_MODE
**Type:** Boolean  
**Default:** `false`  
**Recommended:** `true` (officially recommended in LMI v16)  
**Example:** `"true"`  
**Description:** Enable async mode (officially recommended for vLLM 0.10.2+). Integrates with vLLM AsyncLLMEngine for improved request handling. Supports custom input/output formatters, multi-adapter (LoRA) serving, and session-based sticky routing.

### OPTION_ROLLING_BATCH
**Type:** String  
**Options:** `"disable"`, `"auto"`, `"vllm"`  
**Default:** `"disable"`  
**Example:** `"disable"` (for async mode), `"vllm"` (for rolling batch mode)  
**Description:** 
- `disable`: No rolling batch (use with async mode)
- `vllm`: Enable vLLM rolling batch
- `auto`: Automatically select

### TENSOR_PARALLEL_DEGREE
**Type:** String/Integer  
**Default:** `"1"`  
**Example:** `"max"`, `"4"`, `"8"`  
**Description:** 
- `max`: Use all available GPUs
- Number: Specific number of GPUs for tensor parallelism

### OPTION_MODEL_LOADING_TIMEOUT
**Type:** Integer  
**Default:** `240` (seconds)  
**Example:** `"1500"`  
**Description:** Timeout for model loading in seconds. Increase for large models.

### SERVING_FAIL_FAST
**Type:** Boolean  
**Default:** `false`  
**Example:** `"true"`  
**Description:** Fail immediately if model loading fails

---

## Performance Tuning

### GPU_MEMORY_UTILIZATION
**Type:** Float  
**Default:** `0.9`  
**Range:** `0.0` - `1.0`  
**Example:** `"0.95"`  
**Description:** Fraction of GPU memory to use. Reduce if OOM occurs.

### OPTION_MAX_MODEL_LEN
**Type:** Integer  
**Default:** Model's max length  
**Example:** `"4096"`, `"8192"`  
**Description:** Maximum sequence length (context window)

### OPTION_MAX_NUM_BATCHED_TOKENS
**Type:** Integer  
**Default:** Model dependent  
**Example:** `"32768"`  
**Description:** Maximum tokens processed in single batch iteration (prefill + decode)

### OPTION_MAX_NUM_SEQS
**Type:** Integer  
**Default:** `256`  
**Example:** `"128"`  
**Description:** Maximum number of sequences per iteration

### OPTION_MAX_ROLLING_BATCH_SIZE
**Type:** Integer  
**Default:** `32`  
**Example:** `"64"`  
**Description:** Maximum batch size for rolling batch mode

### OPTION_ENABLE_CHUNKED_PREFILL
**Type:** Boolean  
**Default:** `false`  
**Example:** `"true"`  
**Description:** Enable chunked prefill for better throughput

### OPTION_ENABLE_PREFIX_CACHING
**Type:** Boolean  
**Default:** `false`  
**Example:** `"true"`  
**Description:** Enable prefix caching (automatic prompt caching)

### OPTION_CPU_OFFLOAD_GB_PER_GPU
**Type:** Float  
**Default:** `0`  
**Example:** `"4"`  
**Description:** CPU offload space in GiB per GPU. Use for large models.

---

## Quantization

### OPTION_QUANTIZE
**Type:** String  
**Options:** `"awq"`, `"gptq"`, `"fp8"`, `"bitsandbytes"`, `"squeezellm"`  
**Default:** None  
**Example:** `"fp8"`  
**Description:** 
- **Runtime quantization:** `fp8`, `bitsandbytes`
- **Pre-quantized models:** Leave empty (vLLM auto-detects)
- **Important:** Don't set for pre-quantized models to enable Marlin optimizations

**Quantization Best Practices:**
- Pre-quantize models when possible
- For pre-quantized models, omit `OPTION_QUANTIZE`
- vLLM will auto-detect and optimize (e.g., use Marlin kernels for AWQ/GPTQ/FP8)

---

## LoRA Adapters

### OPTION_ENABLE_LORA
**Type:** Boolean  
**Default:** `false`  
**Example:** `"true"`  
**Description:** Enable LoRA adapter support

### OPTION_MAX_LORAS
**Type:** Integer  
**Default:** `1`  
**Example:** `"4"`  
**Description:** Maximum number of LoRA adapters

### OPTION_MAX_LORA_RANK
**Type:** Integer  
**Default:** `16`  
**Example:** `"64"`  
**Description:** Maximum LoRA rank

### OPTION_LORA_EXTRA_VOCAB_SIZE
**Type:** Integer  
**Default:** `256`  
**Example:** `"512"`  
**Description:** Extra vocabulary size for LoRA

### OPTION_MAX_CPU_LORAS
**Type:** Integer  
**Default:** None  
**Example:** `"8"`  
**Description:** Maximum LoRA adapters in CPU memory

---

## Advanced Configuration

### OPTION_TOKENIZER_MODE
**Type:** String  
**Options:** `"auto"`, `"slow"`, `"mistral"`  
**Default:** `"auto"`  
**Example:** `"mistral"`  
**Description:** Tokenizer mode selection

### OPTION_TRUST_REMOTE_CODE
**Type:** Boolean  
**Default:** `false`  
**Example:** `"true"`  
**Description:** Trust remote code from HuggingFace

### OPTION_DTYPE
**Type:** String  
**Options:** `"auto"`, `"float16"`, `"bfloat16"`, `"float32"`  
**Default:** `"auto"`  
**Example:** `"bfloat16"`  
**Description:** Data type for model weights

### OPTION_SEED
**Type:** Integer  
**Default:** `0`  
**Example:** `"42"`  
**Description:** Random seed for reproducibility

### OPTION_SWAP_SPACE
**Type:** Integer  
**Default:** `4` (GB)  
**Example:** `"8"`  
**Description:** CPU swap space in GiB

### OPTION_ENFORCE_EAGER
**Type:** Boolean  
**Default:** `false`  
**Example:** `"true"`  
**Description:** Disable CUDA graph (use for debugging)

### OPTION_DISABLE_CUSTOM_ALL_REDUCE
**Type:** Boolean  
**Default:** `false`  
**Example:** `"true"`  
**Description:** Disable custom all-reduce kernel

---

## vLLM Engine Arguments

All vLLM [EngineArguments](https://docs.vllm.ai/en/v0.10.2/serving/engine_args.html) are supported as pass-through configurations.

### Format
- **Environment Variable:** `OPTION_<ARGUMENT_NAME>` (uppercase with underscores)
- **serving.properties:** `option.<argument_name>` (lowercase with underscores)

### Examples

#### Boolean Configuration
```bash
# Enable prefix caching
OPTION_ENABLE_PREFIX_CACHING=true
```

#### String Configuration
```bash
# Set tokenizer mode
OPTION_TOKENIZER_MODE=mistral
```

#### JSON/Object Configuration
```bash
# Speculative decoding
OPTION_SPECULATIVE_CONFIG='{"model": "meta-llama/Llama3.2-1B-Instruct", "num_speculative_tokens": 5}'
```

---

## Troubleshooting

### Out of Memory (OOM)
1. Reduce `GPU_MEMORY_UTILIZATION` (try 0.85 or 0.8)
2. Reduce `OPTION_MAX_MODEL_LEN`
3. Reduce `OPTION_MAX_NUM_BATCHED_TOKENS`
4. Enable `OPTION_CPU_OFFLOAD_GB_PER_GPU`
5. Increase `TENSOR_PARALLEL_DEGREE`

### Slow Model Loading
1. Increase `OPTION_MODEL_LOADING_TIMEOUT`
2. Check network connectivity to HuggingFace
3. Consider using local model path instead of HF_MODEL_ID

### Poor Performance
1. Enable `OPTION_ENABLE_PREFIX_CACHING`
2. Enable `OPTION_ENABLE_CHUNKED_PREFILL`
3. Increase `GPU_MEMORY_UTILIZATION` (if not OOM)
4. Tune `OPTION_MAX_NUM_BATCHED_TOKENS`
5. For pre-quantized models, remove `OPTION_QUANTIZE`

### Quantization Issues
1. For pre-quantized models, **do not** set `OPTION_QUANTIZE`
2. vLLM will auto-detect and optimize (e.g., use Marlin kernels)
3. Only use `OPTION_QUANTIZE` for runtime quantization (fp8, bitsandbytes)

---

## References

- [DJL LMI vLLM User Guide](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/vllm_user_guide.html)
- [LMI v16 Release Notes](https://docs.djl.ai/master/docs/serving/serving/docs/lmi/release_notes.html#lmi-v16-djl-serving-0340)
- [vLLM Engine Arguments](https://docs.vllm.ai/en/v0.10.2/serving/engine_args.html)
- [vLLM Supported Models](https://docs.vllm.ai/en/v0.10.2/models/supported_models.html)
- [vLLM Quantization](https://docs.vllm.ai/en/v0.10.2/quantization/supported_hardware.html)

## LMI v16 New Features

LMI v16 includes:
- vLLM upgraded to 0.10.2
- Async mode is now officially recommended
- Async vLLM handler supports custom input/output formatters
- Async vLLM handler supports multi-adapter (LoRA) serving
- Async vLLM handler supports session-based sticky routing

**Container Image:**
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16.0.0-cu128
```
