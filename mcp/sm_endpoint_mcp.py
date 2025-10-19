#!/usr/bin/env python3

import os
import json
import time
import asyncio
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List
import boto3
from fastmcp import FastMCP
from datasets import load_dataset

# Initialize MCP server
mcp = FastMCP("SageMaker Endpoint Tools")

def get_sagemaker_client(region=None):
    """Get SageMaker client"""
    region = region or os.getenv('AWS_REGION', 'us-east-1')
    return boto3.client('sagemaker', region_name=region)

def get_cloudwatch_client(region=None):
    """Get CloudWatch client"""
    region = region or os.getenv('AWS_REGION', 'us-east-1')
    return boto3.client('cloudwatch', region_name=region)

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
        runtime = boto3.client('sagemaker-runtime', region_name=region)
        
        # Default test payload if none provided
        if not test_payload:
            test_payload = json.dumps({"inputs": "Hello, how are you?"})
        
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
        end_time = datetime.utcnow()
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
        runtime = boto3.client('sagemaker-runtime', region_name=region)
        
        # Load dataset if specified
        test_inputs = []
        if dataset_name:
            try:
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
            # Create payload with max_new_tokens=100 for output length control
            payload = json.dumps({
                "inputs": input_text,
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.7,
                    "do_sample": True
                }
            })
            start_time = time.time()
            
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=payload
            )
            
            end_time = time.time()
            response_body = response['Body'].read().decode('utf-8')
            
            return {
                'latency': (end_time - start_time) * 1000,
                'success': True,
                'response_size': len(response_body),
                'input_length': len(input_text)
            }
        
        # Run benchmark
        start_time = time.time()
        results = []
        
        for i in range(num_requests):
            try:
                input_text = test_inputs[i] if i < len(test_inputs) else test_inputs[0]
                result = send_request(input_text)
                results.append(result)
            except Exception as e:
                results.append({'latency': 0, 'success': False, 'error': str(e)})
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        successful_requests = [r for r in results if r['success']]
        latencies = [r['latency'] for r in successful_requests]
        input_lengths = [r.get('input_length', 0) for r in successful_requests]
        
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
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
            'requests_per_second': round(num_requests / total_time, 2),
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
