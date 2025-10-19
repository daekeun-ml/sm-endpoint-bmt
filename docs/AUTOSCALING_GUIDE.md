# Auto Scaling Guide

A comprehensive guide for configuring and using SageMaker Endpoint Auto Scaling.

## Table of Contents

- [Overview](#overview)
- [CloudWatch Metrics](#cloudwatch-metrics)
- [Auto Scaling Configuration](#auto-scaling-configuration)
- [Test Scenarios](#test-scenarios)
- [Monitoring](#monitoring)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

SageMaker Endpoint Auto Scaling automatically adjusts the number of instances based on CloudWatch metrics.

### Key Features

- **Scale Out**: Add instances when traffic increases
- **Scale In**: Remove instances when traffic decreases
- **Metrics-based**: Scaling decisions based on CloudWatch metrics
- **Cooldown Period**: Prevents rapid scaling fluctuations

### How It Works

```
High Traffic → CloudWatch Metric ↑ → Scale Out → More Instances
Low Traffic  → CloudWatch Metric ↓ → Scale In  → Fewer Instances
```

---

## CloudWatch Metrics

### 1. InvocationsPerInstance (Predefined Metric)

**Type:** Predefined Metric  
**Description:** Number of requests per instance (average per minute)

**When to Use:**
- General scaling scenarios
- Request count-based scaling
- Stable traffic patterns

**Recommended Values:**
- Small models (< 10B): 1000-2000
- Medium models (10B-70B): 500-1000
- Large models (> 70B): 100-500

**Example:**
```bash
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --target-invocations-per-instance 1000
```

### 2. ConcurrentRequestsPerModel (High-Resolution Predefined Metric)

**Type:** High-Resolution Predefined Metric (10-second intervals)  
**Description:** Number of concurrent requests per model

**Key Features:**
- High-resolution metric collected every 10 seconds
- Up to 6x faster scale-up detection
- Optimized for Generative AI models
- Sub-minute scaling response

**When to Use:**
- When concurrency is critical
- Real-time response requirements
- Rapid response to traffic spikes
- LLM and Foundation Models

**Recommended Values:**
- Small models: 50-100
- Medium models: 20-50
- Large models: 5-20

**Example:**
```bash
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --target-concurrent-requests-per-model 50
```

**Reference:**
- AWS Blog: [Faster Auto Scaling for Generative AI Models](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-inference-launches-faster-auto-scaling-for-generative-ai-models/)
- Available on accelerated instances (g4dn, g5, g6, p2, p3, p4d, p4de, p5, inf1, inf2, trn1n, trn1)

### 3. ConcurrentRequestsPerCopy (High-Resolution Predefined Metric)

**Type:** High-Resolution Predefined Metric (10-second intervals)  
**Description:** Number of concurrent requests per copy of an Inference Component

**What are Inference Components?**
- Feature to deploy multiple models on a single endpoint
- Each model is deployed as an independent Inference Component
- Fine-grained resource allocation (CPU, GPU, memory)

**What is a Copy?**
- Runtime copy of a model container
- Number specified by `CopyCount` parameter
- Each copy processes inference requests independently

**Example Structure:**
```
Endpoint
├── Inference Component A (Llama-70B)
│   ├── Copy 1 (GPU 0-3)
│   └── Copy 2 (GPU 4-7)
└── Inference Component B (Llama-8B)
    ├── Copy 1 (GPU 0)
    ├── Copy 2 (GPU 1)
    └── Copy 3 (GPU 2)
```

**Key Features:**
- High-resolution metric collected every 10 seconds
- Automatically adjusts the number of copies in Inference Components
- Efficient hosting of multiple models

**When to Use:**
- When using Inference Components (required)
- Deploying multiple models on one endpoint
- Independent scaling per model needed
- Resource efficiency optimization

**Recommended Values:**
- Generally 5-20
- Adjust based on model size and processing time

**Example:**
```bash
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --target-concurrent-requests-per-copy 10
```

**Note:**
- Not used for regular endpoints without Inference Components
- Use `ConcurrentRequestsPerModel` instead
- [Inference Components Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html)

### Metrics Comparison

| Metric | Type | Collection Interval | Use Case | Scaling Speed |
|--------|------|---------------------|----------|---------------|
| InvocationsPerInstance | Predefined | 1 minute | General Endpoint (single model) | Standard |
| ConcurrentRequestsPerModel | High-Resolution | 10 seconds | General Endpoint (LLM, real-time) | Up to 6x faster |
| ConcurrentRequestsPerCopy | High-Resolution | 10 seconds | Inference Components (multi-model) | Up to 6x faster |

### Hybrid Approach (Recommended)

You can use multiple metrics simultaneously for more efficient scaling:

```bash
# InvocationsPerInstance + ConcurrentRequestsPerModel combination
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 1 \
  --max-capacity 10 \
  --target-invocations-per-instance 1000 \
  --target-concurrent-requests-per-model 50
```

**Benefits:**
- Sudden traffic spikes: ConcurrentRequestsPerModel responds quickly
- Gradual traffic increases: InvocationsPerInstance provides stable response
- Container failures: Both metrics complement each other

---

## Auto Scaling Configuration

### Basic Configuration

```bash
# 1. Configure Auto Scaling
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 1 \
  --max-capacity 10 \
  --target-invocations-per-instance 1000

# 2. Check configuration
uv run python autoscaling/autoscaling.py describe --endpoint-name "my-endpoint"

# 3. Check metrics
uv run python autoscaling/autoscaling.py metrics --endpoint-name "my-endpoint"
```

### Advanced Configuration (Multiple Metrics)

```bash
# Use all 3 metrics (hybrid approach)
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 2 \
  --max-capacity 20 \
  --target-invocations-per-instance 1000 \
  --target-concurrent-requests-per-model 50 \
  --target-concurrent-requests-per-copy 10 \
  --scale-in-cooldown 300 \
  --scale-out-cooldown 60
```

**How it works:**
- 3 independent Auto Scaling policies are created
- Each policy monitors its metric independently
- Scale Out triggered when any policy exceeds threshold
- Scale In occurs when all policy metrics are below target

**Recommended Combinations:**
1. **LLM Workloads:**
   ```bash
   --target-invocations-per-instance 500 \
   --target-concurrent-requests-per-model 30
   ```

2. **Real-time Inference:**
   ```bash
   --target-concurrent-requests-per-model 50 \
   --target-concurrent-requests-per-copy 10
   ```

3. **Stable Workloads:**
   ```bash
   --target-invocations-per-instance 1000
   ```

### Parameter Description

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `min-capacity` | Minimum number of instances | 1 | 1-2 |
| `max-capacity` | Maximum number of instances | 10 | 1.5x expected peak load |
| `scale-in-cooldown` | Scale In wait time (seconds) | 300 | 300-600 |
| `scale-out-cooldown` | Scale Out wait time (seconds) | 60 | 30-60 |

---

## Test Scenarios

### 1. Full Cycle Test (Recommended)

Tests the complete Auto Scaling cycle.

```bash
uv run python autoscaling/test_autoscaling.py \
  --endpoint-name "my-endpoint" \
  --test full-cycle \
  --num-requests 100 \
  --duration 300 \
  --scale-in-wait 600 \
  --metrics-interval 30
```

**Test Flow:**
1. Check initial state
2. Generate high load (5 minutes)
3. Verify Scale Out
4. Stop load
5. Verify Scale In

**Expected Results:**
```
Initial Instances: 1
After Scale Out: 3-5
After Scale In: 1
```

### 2. Scale Out Test

Tests Scale Out only.

```bash
uv run python autoscaling/test_autoscaling.py \
  --endpoint-name "my-endpoint" \
  --test scale-out \
  --num-requests 100 \
  --duration 300
```

**Test Flow:**
1. Check initial state
2. Generate high load
3. Monitor CloudWatch metrics
4. Verify instance increase

**Success Criteria:**
- Metrics exceed target values
- Instance count increases
- Requests processed successfully

### 3. Scale In Test

Tests Scale In only.

```bash
uv run python autoscaling/test_autoscaling.py \
  --endpoint-name "my-endpoint" \
  --test scale-in \
  --scale-in-wait 600
```

**Test Flow:**
1. Check current state (multiple instances)
2. Wait without load
3. Wait for cooldown period
4. Verify instance decrease

**Success Criteria:**
- Metrics below target values
- Instance decrease after cooldown period

---

## Monitoring

### Check CloudWatch Metrics

```bash
# Last 5 minutes metrics
uv run python autoscaling/autoscaling.py metrics --endpoint-name "my-endpoint"

# Last 30 minutes metrics
uv run python autoscaling/autoscaling.py metrics --endpoint-name "my-endpoint" --period 30
```

### AWS Console Monitoring

1. **CloudWatch Dashboard**
   ```
   https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:
   ```

2. **SageMaker Endpoint Monitoring**
   ```
   https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/endpoints
   ```

3. **Auto Scaling Policies**
   ```
   https://console.aws.amazon.com/ec2autoscaling/home?region=us-east-1#/details
   ```

### Key Monitoring Points

- **Current Instance Count**: Current number of instances
- **Desired Instance Count**: Target number of instances
- **InvocationsPerInstance**: Requests per instance
- **ConcurrentRequests**: Concurrent requests
- **Model Latency**: Model response time

---

## Best Practices

### 1. Initial Setup

```bash
# Conservative start
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 1 \
  --max-capacity 5 \
  --target-invocations-per-instance 500 \
  --scale-in-cooldown 600 \
  --scale-out-cooldown 60
```

**Rationale:**
- Low target value for quick scaling
- Long Scale In cooldown for stability
- Small max-capacity for cost control

### 2. Production Setup

```bash
# After analyzing actual traffic patterns
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 2 \
  --max-capacity 20 \
  --target-invocations-per-instance 1000 \
  --scale-in-cooldown 300 \
  --scale-out-cooldown 60
```

**Rationale:**
- min-capacity 2 for high availability
- Target values based on actual traffic
- Appropriate max-capacity for scalability

### 3. Cost Optimization

```bash
# Cost-focused
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 1 \
  --max-capacity 10 \
  --target-invocations-per-instance 1500 \
  --scale-in-cooldown 180 \
  --scale-out-cooldown 60
```

**Rationale:**
- High target value to minimize instances
- Short Scale In cooldown for quick reduction
- min-capacity 1 to reduce idle time costs

### 4. Performance Optimization

```bash
# Performance-focused
uv run python autoscaling/autoscaling.py configure \
  --endpoint-name "my-endpoint" \
  --min-capacity 3 \
  --max-capacity 30 \
  --target-invocations-per-instance 500 \
  --target-concurrent-requests-per-model 30 \
  --scale-in-cooldown 600 \
  --scale-out-cooldown 30
```

**Rationale:**
- Low target values for headroom
- Fast Scale Out to minimize latency
- Multiple metrics for fine-grained control

---

## Troubleshooting

### Slow Scale Out

**Symptoms:**
- Instances don't increase despite traffic growth
- Response times become longer

**Solutions:**
1. Lower target values
   ```bash
   --target-invocations-per-instance 500  # Reduced from 1000
   ```

2. Reduce Scale Out cooldown
   ```bash
   --scale-out-cooldown 30  # Reduced from 60
   ```

3. Check metrics
   ```bash
   uv run python autoscaling/autoscaling.py metrics --endpoint-name "my-endpoint"
   ```

### Scale In Too Fast

**Symptoms:**
- Instances decrease with slight traffic reduction
- Latency when traffic increases again

**Solutions:**
1. Increase Scale In cooldown
   ```bash
   --scale-in-cooldown 600  # Increased from 300
   ```

2. Adjust target values
   ```bash
   --target-invocations-per-instance 800  # Add headroom
   ```

### Instances Reach max-capacity

**Symptoms:**
- Consistently at max-capacity
- Response times still long

**Solutions:**
1. Increase max-capacity
   ```bash
   --max-capacity 20  # Increased from 10
   ```

2. Review target values
   ```bash
   # Check current metrics
   uv run python autoscaling/autoscaling.py metrics --endpoint-name "my-endpoint"
   
   # Adjust to appropriate target
   --target-invocations-per-instance 1200
   ```

3. Upgrade instance type
   ```bash
   # Use larger instances
   --instance-type "ml.g5.12xlarge"
   ```

### CloudWatch Metrics Not Visible

**Symptoms:**
- No data when running `uv run python autoscaling/autoscaling.py metrics`

**Solutions:**
1. Send requests to endpoint
   ```bash
   python test_endpoint.py --endpoint-name "my-endpoint"
   ```

2. Wait (1-2 minutes)

3. Check again
   ```bash
   uv run python autoscaling/autoscaling.py metrics --endpoint-name "my-endpoint"
   ```

### Auto Scaling Policy Not Applied

**Symptoms:**
- Configured but scaling doesn't work

**Solutions:**
1. Check configuration
   ```bash
   uv run python autoscaling/autoscaling.py describe --endpoint-name "my-endpoint"
   ```

2. Check IAM permissions
   - `application-autoscaling:*` permission required
   - `cloudwatch:GetMetricStatistics` permission required

3. Reconfigure
   ```bash
   # Delete and reconfigure
   uv run python autoscaling/autoscaling.py delete --endpoint-name "my-endpoint"
   uv run python autoscaling/autoscaling.py configure --endpoint-name "my-endpoint" ...
   ```

---

## References

- [AWS SageMaker Auto Scaling](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling.html)
- [CloudWatch Metrics for SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html)
- [Application Auto Scaling](https://docs.aws.amazon.com/autoscaling/application/userguide/what-is-application-auto-scaling.html)