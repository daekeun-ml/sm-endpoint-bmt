#!/usr/bin/env python3

import os
import json
import time
import asyncio
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
import boto3
from fastmcp import FastMCP

# 🔴 datasets 는 최상단에서 import 하지 않는다. 이 파일에서 쓰는 곳은 dataset_name 을 준
#    벤치마크 한 곳뿐인데, 최상단에 두면 서버를 띄우는 것만으로도 설치를 강제한다
#    (pyproject 의 mcp extra 에는 fastmcp 만 있어 실제로 ModuleNotFoundError 가 났다).
#    쓰는 자리에서 import 하고, 없으면 무엇을 설치해야 하는지 알려 준다.

# Initialize MCP server
mcp = FastMCP("SageMaker Endpoint Tools")

def _session(region=None):
    """
    Build an explicit boto3 Session.

    boto3.client() is an alias for the shared DEFAULT session, and the boto3 docs
    warn that calling it from a concurrent context can cause response ordering
    issues or SSL interpreter failures. An MCP server serves concurrent tool calls,
    which is exactly that hazard.
    """
    return boto3.session.Session(
        region_name=region or os.getenv('AWS_REGION', 'us-east-1')
    )

def get_sagemaker_client(region=None):
    """Get SageMaker client"""
    return _session(region).client('sagemaker')

def get_cloudwatch_client(region=None):
    """Get CloudWatch client"""
    return _session(region).client('cloudwatch')

def get_runtime_client(region=None, max_pool_connections=10):
    """
    Get a sagemaker-runtime client sized for the requested concurrency.

    Reusing one client also preserves keep-alive, so measured latency does not
    include a fresh TLS handshake on every call.
    """
    from botocore.config import Config

    return _session(region).client(
        'sagemaker-runtime',
        config=Config(
            max_pool_connections=max(max_pool_connections, 10),
            retries={'mode': 'standard', 'total_max_attempts': 1},
            tcp_keepalive=True,
        ),
    )

@mcp.tool()
def test_endpoint(endpoint_name: str, test_payload: str = None, region: str = None) -> Dict[str, Any]:
    """
    Test SageMaker endpoint with a sample request
    
    Args:
        endpoint_name: SageMaker endpoint name
        test_payload: JSON payload to send (optional, uses default if not provided)
        region: AWS region (optional)
    
    Returns:
        Test result with response and timing
    """
    try:
        region = region or os.getenv('AWS_REGION', 'us-east-1')
        runtime = get_runtime_client(region)
        
        # Default test payload if none provided.
        # Use the OpenAI /v1/completions schema that the vLLM/LMI container actually
        # serves. The TGI-style {"inputs", "parameters": {"max_new_tokens"}} body is
        # silently ignored by a vLLM OpenAI endpoint, so max_new_tokens never caps
        # the output and the request falls back to the server default length.
        if not test_payload:
            test_payload = json.dumps({
                "prompt": "Hello, how are you?",
                "max_tokens": 100,
            })
        
        start_time = time.time()
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=test_payload
        )
        end_time = time.time()
        
        response_body = response['Body'].read().decode('utf-8')
        
        return {
            'success': True,
            'endpoint_name': endpoint_name,
            'response_time_ms': round((end_time - start_time) * 1000, 2),
            'response': response_body,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'endpoint_name': endpoint_name,
            'timestamp': datetime.now().isoformat()
        }

@mcp.tool()
def check_endpoint_metrics(endpoint_name: str, minutes: int = 30, region: str = None) -> Dict[str, Any]:
    """
    Check CloudWatch metrics for SageMaker endpoint
    
    Args:
        endpoint_name: SageMaker endpoint name
        minutes: Time range in minutes to check (default: 30)
        region: AWS region (optional)
    
    Returns:
        Endpoint metrics summary
    """
    try:
        cloudwatch = get_cloudwatch_client(region)
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=minutes)
        
        metrics = {}
        metric_names = ['Invocations', 'Invocation4XXErrors', 'Invocation5XXErrors', 'ModelLatency']
        
        for metric_name in metric_names:
            response = cloudwatch.get_metric_statistics(
                Namespace='AWS/SageMaker',
                MetricName=metric_name,
                Dimensions=[{'Name': 'EndpointName', 'Value': endpoint_name}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Sum', 'Average']
            )
            
            if response['Datapoints']:
                latest = max(response['Datapoints'], key=lambda x: x['Timestamp'])
                metrics[metric_name] = {
                    'value': latest.get('Sum', latest.get('Average', 0)),
                    'timestamp': latest['Timestamp'].isoformat()
                }
            else:
                metrics[metric_name] = {'value': 0, 'timestamp': None}
        
        return {
            'success': True,
            'endpoint_name': endpoint_name,
            'time_range_minutes': minutes,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'endpoint_name': endpoint_name,
            'timestamp': datetime.now().isoformat()
        }

@mcp.tool()
def run_benchmark(endpoint_name: str, num_requests: int = 10, concurrent_requests: int = 1, region: str = None, dataset_name: str = None) -> Dict[str, Any]:
    """
    Run performance benchmark on SageMaker endpoint
    
    Args:
        endpoint_name: SageMaker endpoint name
        num_requests: Total number of requests to send (default: 10)
        concurrent_requests: Number of concurrent requests (default: 1)
        region: AWS region (optional)
        dataset_name: HuggingFace dataset name (e.g., 'tatsu-lab/alpaca')
    
    Returns:
        Benchmark results with performance metrics
    """
    try:
        region = region or os.getenv('AWS_REGION', 'us-east-1')
        # The connection pool must be at least as large as the concurrency, or
        # connections above the default 10 get re-handshaked and discarded.
        runtime = get_runtime_client(region, max_pool_connections=concurrent_requests)
        
        # Load dataset if specified
        test_inputs = []
        if dataset_name:
            try:
                try:
                    from datasets import load_dataset
                except ImportError as e:
                    raise RuntimeError(
                        f"dataset_name='{dataset_name}' 을 쓰려면 datasets 가 필요합니다: "
                        "pip install 'sm-endpoint-bmt[datasets]' "
                        "(dataset_name 을 비우면 내장 프롬프트로 돕니다)") from e
                dataset = load_dataset(dataset_name, split='train')
                # Sample random entries from the dataset
                sample_size = min(num_requests, len(dataset))
                sample_indices = random.sample(range(len(dataset)), sample_size)
                
                for idx in sample_indices:
                    item = dataset[idx]
                    # Try different common field names for instruction/input
                    if 'instruction' in item:
                        text = item['instruction']
                        if 'input' in item and item['input']:
                            text += f"\n\nInput: {item['input']}"
                    elif 'text' in item:
                        text = item['text']
                    elif 'prompt' in item:
                        text = item['prompt']
                    else:
                        # Use the first string field found
                        text = next((v for v in item.values() if isinstance(v, str)), "Test message")
                    
                    # Limit input length to approximately 1024 tokens (roughly 4 chars per token)
                    if len(text) > 4000:
                        text = text[:4000]
                    
                    test_inputs.append(text)
                    
            except Exception as e:
                print(f"Failed to load dataset {dataset_name}: {e}")
                test_inputs = ["Test message for benchmarking"] * num_requests
        else:
            test_inputs = ["Test message for benchmarking"] * num_requests
        
        def send_request(input_text):
            # OpenAI /v1/completions schema: max_tokens is the key a vLLM/LMI
            # container honours. max_new_tokens is TGI-only and is silently dropped.
            payload = json.dumps({
                "prompt": input_text,
                "max_tokens": 100,
                "temperature": 0.7,
            })
            # perf_counter, not time.time: wall clock is NTP-adjustable and a clock
            # step mid-run would corrupt the latency numbers.
            start = time.perf_counter()

            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=payload
            )

            response_body = response['Body'].read().decode('utf-8')
            elapsed = time.perf_counter() - start

            return {
                'latency': elapsed * 1000,
                'success': True,
                'response_size': len(response_body),
                'input_length': len(input_text)
            }

        def send_request_safe(input_text):
            try:
                return send_request(input_text)
            except Exception as e:
                return {'latency': 0, 'success': False, 'error': str(e)}

        # Run benchmark. concurrent_requests used to be accepted and then ignored -
        # every request went out sequentially, so requests_per_second was 1/latency
        # by construction no matter what the caller asked for.
        start_time = time.perf_counter()
        inputs = [
            test_inputs[i] if i < len(test_inputs) else test_inputs[0]
            for i in range(num_requests)
        ]
        if concurrent_requests > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=concurrent_requests) as pool:
                results = list(pool.map(send_request_safe, inputs))
        else:
            results = [send_request_safe(text) for text in inputs]

        total_time = time.perf_counter() - start_time
        
        # Calculate statistics
        successful_requests = [r for r in results if r['success']]
        latencies = [r['latency'] for r in successful_requests]
        input_lengths = [r.get('input_length', 0) for r in successful_requests]
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            # min(..., n-1) guards the nearest-rank IndexError when
            # len(latencies) * 0.95 lands exactly on n. numpy's linear
            # interpolation (what vllm bench serve uses) is preferred when
            # available so the number is comparable.
            try:
                import numpy as _np

                p95_latency = float(_np.percentile(latencies, 95))
            except ImportError:
                idx = min(int(len(latencies) * 0.95), len(latencies) - 1)
                p95_latency = sorted(latencies)[idx]
            avg_input_length = sum(input_lengths) / len(input_lengths) if input_lengths else 0
        else:
            avg_latency = min_latency = max_latency = p95_latency = avg_input_length = 0
        
        return {
            'success': True,
            'endpoint_name': endpoint_name,
            'dataset_used': dataset_name or 'default',
            'total_requests': num_requests,
            'successful_requests': len(successful_requests),
            'failed_requests': num_requests - len(successful_requests),
            'total_time_seconds': round(total_time, 2),
            # Successful requests over the measured window, matching
            # vllm bench serve's completed/dur_s. Counting failures here would
            # inflate throughput on a broken endpoint.
            'requests_per_second': round(len(successful_requests) / total_time, 2)
            if total_time > 0 else 0,
            'concurrent_requests': concurrent_requests,
            'latency_stats': {
                'average_ms': round(avg_latency, 2),
                'min_ms': round(min_latency, 2),
                'max_ms': round(max_latency, 2),
                'p95_ms': round(p95_latency, 2)
            },
            'input_stats': {
                'average_input_length': round(avg_input_length, 2)
            },
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'endpoint_name': endpoint_name,
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    mcp.run()
