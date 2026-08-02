# SageMaker Endpoint Benchmark Tool - Usage Guide

The CLI mirrors `vllm bench serve`, so a flag that works there works here. This guide lists every
argument and groups them the way `--help` does.

## Quick Start

```bash
# Smallest useful run: verify the endpoint answers before measuring anything
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --num-prompts 10

# A real measurement, saved to JSON
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --num-prompts 200 \
  --max-concurrency 20 \
  --save-result
```

## Command Line Arguments

Defaults below are the argparse defaults. Where a default is `None`, the flag is omitted from the
request and the container decides.

### Endpoint

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--endpoint-name` | string | required | SageMaker endpoint name |
| `--region` | string | `us-east-1` | AWS region |
| `--endpoint-url` | string | None | SageMaker Runtime API URL override. Used for local verification against a proxy |
| `--endpoint-type` / `--backend` | choice | `openai` | Payload schema: `openai` (`/v1/completions` style) or `openai-chat` (messages). Also accepts `completions`, `chat` |
| `--model` | string | None | Model id sent in the payload. Most LMI containers ignore it; set it when one container serves several models |
| `--tokenizer` | string | None | HuggingFace tokenizer for token-accurate random prompts and as the output-token fallback |
| `--label` | string | None | Label recorded in the result file name (falls back to `sagemaker`) |

### SageMaker Transport

These exist because the AWS boundary behaves differently from plain HTTP.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--read-timeout` | int | 900 | botocore per-socket-read timeout in seconds. The botocore default of 60s cuts long generations |
| `--connect-timeout` | int | 10 | botocore connect timeout in seconds |
| `--no-include-usage` | flag | False | Do not send `stream_options.include_usage`. Only for containers that reject the field. Without usage, output token counts become estimates |

### Dataset

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset-name` | string | `random` | `random`, `sharegpt`, `huggingface`, `hf` |
| `--dataset-path` | string | None | File path (sharegpt) or HuggingFace dataset ID |
| `--num-prompts` | int | 200 | Number of prompts to send |
| `--disable-shuffle` | flag | False | Sample sequentially instead of shuffling |
| `--seed` | int | 0 | Random seed for sampling |

### HuggingFace Dataset

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--hf-prompt-column` | string | `prompt` | Column holding prompts |
| `--hf-completion-column` | string | `completion` | Column holding completions |

### Random Dataset

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--random-input-len` | int | 1024 | Input length in tokens |
| `--random-output-len` | int | 128 | Output length in tokens |
| `--random-range-ratio` | float | 0.0 | Sampling range: `len * (1 ± ratio)` |
| `--random-prefix-len` | int | 0 | Fixed prefix tokens prepended to every prompt |

### Benchmark

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max-concurrency` | int | None | Cap on in-flight requests. Unset means no cap |
| `--request-rate` | float | `inf` | Requests per second. `inf` sends everything at once |
| `--burstiness` | float | 1.0 | Gamma shape of the arrival process. Only applies when `--request-rate` is finite. 1.0 is Poisson |
| `--num-warmups` | int | 0 | Warmup requests, excluded from all metrics |
| `--ready-check-timeout-sec` | int | 0 | Wait this long for the endpoint to become ready. 0 skips the check |
| `--ignore-eos` | flag | False | Send `ignore_eos` so generation runs to the requested length. Forced on for `--dataset-name random`, matching vLLM |
| `--disable-tqdm` | flag | False | Turn off the progress bar |
| `--percentile-metrics` | string | `ttft,tpot,itl` | Which metrics get percentiles. Allowed: `ttft`, `tpot`, `itl`, `e2el` |
| `--metric-percentiles` | string | `99` | Percentiles to report, e.g. `25,50,75` |
| `--goodput` | KEY:VALUE… | None | SLOs in milliseconds, e.g. `--goodput ttft:200 tpot:50` |
| `--ramp-up-strategy` | choice | None | `linear` or `exponential`. Ramps the rate over the run instead of holding it |
| `--ramp-up-start-rps` | float | None | Starting rate for the ramp |
| `--ramp-up-end-rps` | float | None | Ending rate for the ramp |
| `--request-id-prefix` | string | generated | Prefix for request ids, recorded in the result JSON |

### Results

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--save-result` | flag | False | Write results to JSON |
| `--save-detailed` | flag | False | Include per-request arrays (ttfts, itls, errors) |
| `--append-result` | flag | False | Append to an existing file as JSONL |
| `--result-dir` | string | None | Output directory |
| `--result-filename` | string | None | Output filename. Generated from the run parameters if unset |
| `--metadata` | KEY=VALUE… | None | Extra fields recorded in the JSON, e.g. `tp=8 lmi=26` |

### Sampling Parameters

Every one of these defaults to `None`, which means **the flag is left out of the request** and the
server default applies. This matches `vllm bench serve`.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--temperature` | float | None | Omitted unless set. Pass `0` for greedy |
| `--top-p` | float | None | Nucleus sampling |
| `--top-k` | int | None | Top-k sampling |
| `--min-p` | float | None | Minimum probability threshold |
| `--repetition-penalty` | float | None | >1.0 penalizes repetition |
| `--presence-penalty` | float | None | Encourages new tokens |
| `--frequency-penalty` | float | None | Penalizes frequent tokens |
| `--extra-body` | JSON | None | Merged into the payload, and wins over the flags above |
| `--use-beam-search` | flag | False | Deprecated. Many containers reject it |
| `--best-of` | int | None | Deprecated. Many containers reject it |

> **On temperature**: the tool no longer forces `temperature=0`. Greedy decoding changes when EOS is
> hit, which changes output length, which changes TPOT and throughput. Leaving it to the server is
> what vLLM does, so comparisons line up. Pass `--temperature 0` if you want the old behavior.

## Usage Examples

### 1. Basic random dataset

```bash
uv run python sagemaker_benchmark.py --endpoint-name 'your-endpoint-name'
```

### 2. Custom random dataset

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --dataset-name random \
  --num-prompts 500 \
  --random-input-len 2048 \
  --random-output-len 256
```

### 3. Chat payload schema

Use this when the container serves `/v1/chat/completions` (messages) rather than raw completions.

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --endpoint-type openai-chat \
  --num-prompts 100
```

### 4. ShareGPT dataset

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --dataset-name sharegpt \
  --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000
```

### 5. HuggingFace dataset (Alpaca)

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --dataset-name huggingface \
  --dataset-path "tatsu-lab/alpaca" \
  --hf-prompt-column "instruction" \
  --hf-completion-column "output" \
  --num-prompts 500
```

### 6. Fixed rate with warmups

Warmup requests absorb the first-call cost so it does not land in the percentiles.

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --request-rate 10 \
  --burstiness 1.0 \
  --num-warmups 5 \
  --num-prompts 500
```

### 7. High concurrency

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --num-prompts 1000 \
  --max-concurrency 50
```

### 8. Ramp-up to find the knee

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --ramp-up-strategy linear \
  --ramp-up-start-rps 1 \
  --ramp-up-end-rps 20 \
  --num-prompts 600
```

### 9. Goodput against an SLO

Reports the fraction of requests that met every threshold, not just the average.

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --goodput ttft:200 tpot:50 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,99 \
  --num-prompts 300
```

### 10. Save results for comparison

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --save-result --save-detailed \
  --result-dir ./results \
  --label lmi26-tp4 \
  --metadata tp=4 engine=lmi \
  --num-prompts 200
```

### 11. Deterministic generation

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --temperature 0 \
  --num-prompts 200
```

### 12. Container-specific fields

`--extra-body` merges raw JSON into the payload, for options no flag covers.

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --extra-body '{"stop": ["\n\n"], "seed": 42}' \
  --num-prompts 100
```

## Output Metrics

Definitions follow vLLM exactly, so results sit side by side with a `vllm bench serve` run.

### Request statistics

- **Successful requests** / **Failed requests**
- **Maximum request concurrency**: the `--max-concurrency` cap
- **Peak concurrent requests**: the highest in-flight count actually observed
- **Request rate configured (RPS)**: what was asked for, not what was achieved
- **Benchmark duration**

### Token statistics

- **Total input tokens** / **Total generated tokens**
- **Request throughput** (req/s), **Output token throughput** (tok/s), **Total token throughput**
- **Peak output token throughput**

### Latency metrics

| Metric | Definition |
|---|---|
| **TTFT** | first non-empty chunk arrival − request send |
| **TPOT** | `(latency − ttft) / (output_len − 1)` |
| **ITL** | gaps between consecutive chunks. TTFT is not part of it |
| **E2EL** | request send → last chunk |

Each reports mean, median, std and the percentiles from `--metric-percentiles`.

### SageMaker Specifics

This section is not in vLLM's output. It reports what the AWS boundary adds:

- requests truncated at `finish_reason=length`
- requests that stopped at EOS
- requests where the container sent no usage frame (token counts fall back to estimates)
- an error breakdown by exception type

Truncation is a result, not an error. Counting a cut-off answer as a plain success reports the wrong
latency, so it is broken out separately.

## Tips

1. **Start with `--num-prompts 10`** to confirm the endpoint answers before measuring anything.
2. **Use `--num-warmups`** rather than discarding the first run by hand.
3. **Set `--max-concurrency` to what you intend to serve.** Without it there is no cap and the
   numbers describe a burst, not steady state.
4. **`--request-rate` with `--burstiness`** models realistic arrivals. Unset rate means all at once.
5. **Pass `--temperature 0`** when you need run-to-run reproducibility.
6. **Check the truncation count.** All-truncated with `--dataset-name random` is expected, because
   `--ignore-eos` is forced on there.

## Troubleshooting

### All requests failing

- Check the endpoint name and that the endpoint is `InService`
- Verify AWS credentials and that `--region` matches where the endpoint lives
- Run `uv run python test_endpoint.py` first: it isolates transport from measurement

### High failure rate

- Lower `--max-concurrency`
- Set `--request-rate` to pace the load
- Look at CloudWatch for throttling and `ModelLatency` (in **microseconds**, not milliseconds)

### Latency looks worse than expected

- The botocore connection pool is sized from `--max-concurrency`. If you reach the endpoint through
  your own client instead, the default pool of 10 will queue requests and you will measure the pool
- Raise `--read-timeout` if long generations are being cut

### Token counts look wrong

- `--no-include-usage` makes output counts estimates. Drop the flag if the container accepts usage
- Set `--tokenizer` to the model's tokenizer so input token counts are exact
