# SageMaker Endpoint Benchmark Tool - Usage Guide

## Quick Start

```bash
# Basic benchmark with random dataset
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name'

# With custom parameters
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --num-prompts 500 \
  --max-concurrency 20 \
  --temperature 0.7
```

## Command Line Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--endpoint-name` | string | SageMaker endpoint name (required) |

### Endpoint Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--region` | string | None | AWS region (uses default region if not specified) |

### Dataset Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset-name` | string | `random` | Dataset type: `random`, `sharegpt`, `huggingface`, `hf` |
| `--dataset-path` | string | None | Path to dataset file (sharegpt) or HuggingFace dataset ID (huggingface) |
| `--num-prompts` | int | 200 | Number of prompts to process |
| `--disable-shuffle` | flag | False | Disable shuffling of dataset samples |
| `--seed` | int | 0 | Random seed for dataset sampling |

### HuggingFace Dataset Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hf-prompt-column` | string | `prompt` | Column name for prompts in HuggingFace dataset |
| `--hf-completion-column` | string | `completion` | Column name for completions in HuggingFace dataset |

### Random Dataset Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--random-input-len` | int | 1024 | Random input length in tokens |
| `--random-output-len` | int | 128 | Random output length in tokens |

### Benchmark Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-concurrency` | int | 10 | Maximum number of concurrent requests |
| `--request-rate` | string | `inf` | Request rate in requests/second. Use `inf` for unlimited |

### Sampling Parameters

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--temperature` | float | 0.0 | Sampling temperature (0.0 = greedy, higher = more random) |
| `--top-p` | float | 1.0 | Top-p (nucleus) sampling parameter (0.0 to 1.0) |
| `--top-k` | int | -1 | Top-k sampling parameter (-1 = disabled) |
| `--use-beam-search` | flag | False | Use beam search instead of sampling |
| `--best-of` | int | 1 | Number of sequences to generate and return the best one |
| `--repetition-penalty` | float | 1.0 | Repetition penalty (1.0 = no penalty, >1.0 = penalize repetition) |
| `--presence-penalty` | float | 0.0 | Presence penalty (encourages new tokens) |
| `--frequency-penalty` | float | 0.0 | Frequency penalty (penalizes frequent tokens) |

## Usage Examples

### 1. Basic Random Dataset

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438'
```

### 2. Custom Random Dataset Parameters

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --dataset-name random \
  --num-prompts 500 \
  --random-input-len 2048 \
  --random-output-len 256
```

### 3. ShareGPT Dataset

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --dataset-name sharegpt \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000
```

### 4. HuggingFace Dataset (Alpaca)

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --dataset-name huggingface \
  --dataset-path "tatsu-lab/alpaca" \
  --hf-prompt-column "instruction" \
  --hf-completion-column "output" \
  --num-prompts 500
```

### 5. High Concurrency Test

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --num-prompts 1000 \
  --max-concurrency 50
```

### 6. Rate-Limited Requests

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --request-rate 10 \
  --num-prompts 500
```

### 7. Creative Generation (High Temperature)

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --temperature 0.8 \
  --top-p 0.95 \
  --num-prompts 200
```

### 8. Deterministic Generation (Greedy Decoding)

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --temperature 0.0 \
  --num-prompts 200
```

### 9. Top-k Sampling with Repetition Penalty

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

### 11. Disable Shuffle (Sequential Sampling)

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'gpt-oss-120b-2025-10-16-10-23-39-438' \
  --dataset-name sharegpt \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --disable-shuffle \
  --num-prompts 100
```

## Output Metrics

The benchmark tool provides the following metrics:

### Request Statistics
- **Successful requests**: Number of successfully completed requests
- **Failed requests**: Number of failed requests
- **Maximum request concurrency**: Peak concurrent requests during benchmark
- **Benchmark duration**: Total time taken for the benchmark

### Token Statistics
- **Total input tokens**: Total number of input tokens processed
- **Total generated tokens**: Total number of output tokens generated
- **Request throughput**: Requests per second
- **Output token throughput**: Output tokens per second
- **Total token throughput**: Total tokens (input + output) per second

### Latency Metrics
- **TTFT (Time to First Token)**: Time from request start to first token
  - Mean, Median, P99 in milliseconds
- **TPOT (Time per Output Token)**: Average time per output token (excluding first token)
  - Mean, Median, P99 in milliseconds
- **ITL (Inter-token Latency)**: Time between consecutive tokens
  - Mean, Median, P99 in milliseconds

## Tips

1. **Start with a small number of prompts** to verify your endpoint is working correctly
2. **Use `--temperature 0.0`** for reproducible results
3. **Adjust `--max-concurrency`** based on your endpoint's capacity
4. **Use `--request-rate`** to simulate realistic traffic patterns
5. **Monitor failed requests** - if you see many failures, reduce concurrency or request rate

## Troubleshooting

### All requests failing
- Check your endpoint name is correct
- Verify your AWS credentials are configured
- Ensure the endpoint is in the correct region

### High failure rate
- Reduce `--max-concurrency`
- Add `--request-rate` to limit request rate
- Check endpoint CloudWatch metrics for throttling

### Unexpected metrics
- Verify the endpoint response format matches OpenAI completion API
- Check the test script first: `uv run python test_endpoint.py`
