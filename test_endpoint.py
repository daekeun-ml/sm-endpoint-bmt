"""
SageMaker Endpoint Test Script
Test streaming response using invoke_endpoint_with_response_stream
"""
import argparse
import boto3
import json
import os
import time
from dotenv import load_dotenv

def test_sagemaker_endpoint(endpoint_name: str, region: str = None, debug: bool = False):
    """
    Test SageMaker endpoint with streaming response
    
    Args:
        endpoint_name: SageMaker endpoint name
        region: AWS region (optional)
        debug: Enable debug output (optional)
    """
    
    # Create SageMaker Runtime client
    client = boto3.client('sagemaker-runtime', region_name=region)
    
    # Test request payload
    payload = {
        "prompt": "What is Machine Learning?",
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": True
    }
    
    print(f"Testing endpoint: {endpoint_name}")
    print(f"Request payload: {json.dumps(payload, indent=2)}")
    print("\n" + "="*50)
    print("Streaming response:")
    print("="*50 + "\n")
    
    try:
        start_time = time.time()
        
        # Call invoke_endpoint_with_response_stream
        response = client.invoke_endpoint_with_response_stream(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        # Process streaming response
        event_stream = response['Body']
        full_response = ""
        token_count = 0
        first_token_time = None
        
        for event in event_stream:
            if 'PayloadPart' in event:
                payload_part = event['PayloadPart']['Bytes'].decode('utf-8')
                
                # Record first token time
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = (first_token_time - start_time) * 1000
                    print(f"[TTFT: {ttft:.2f}ms]")
                    
                    if debug:
                        print("\n[DEBUG] First chunk raw data:")
                        print(repr(payload_part[:500]))
                        print()
                
                # Parse and print response
                try:
                    chunk = payload_part.strip()
                    if chunk.startswith('data:'):
                        chunk = chunk[5:].strip()
                    
                    if chunk and chunk != '[DONE]':
                        data = json.loads(chunk)
                        
                        if debug:
                            print(f"[DEBUG] Parsed JSON keys: {list(data.keys())}")
                        
                        if 'token' in data:
                            token = data['token']['text'] if isinstance(data['token'], dict) else data['token']
                            print(token, end='', flush=True)
                            full_response += token
                            token_count += 1
                        elif 'choices' in data and len(data['choices']) > 0:
                            token = data['choices'][0].get('text', '')
                            print(token, end='', flush=True)
                            full_response += token
                            token_count += 1
                        elif 'generated_text' in data:
                            print(f"\n\nGenerated text: {data['generated_text']}")
                        elif 'text' in data:
                            print(data['text'], end='', flush=True)
                            full_response += data['text']
                            token_count += 1
                except json.JSONDecodeError as e:
                    if debug:
                        print(f"[DEBUG] Not JSON: {e}")
                        print(f"[DEBUG] Raw data: {repr(payload_part[:200])}")
                    print(payload_part, end='', flush=True)
                    full_response += payload_part
                    token_count += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n\n" + "="*50)
        print("Test Results:")
        print("="*50)
        print(f"Total time: {total_time:.2f}s")
        print(f"Tokens generated: {token_count}")
        if first_token_time:
            print(f"Time to first token: {ttft:.2f}ms")
        if token_count > 1:
            tpot = ((end_time - first_token_time) / (token_count - 1)) * 1000
            print(f"Time per output token: {tpot:.2f}ms")
        print(f"Throughput: {token_count / total_time:.2f} tokens/s")
        print("\n✅ Endpoint test successful!")
        
    except Exception as e:
        print(f"\n❌ Error testing endpoint: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # Load .env file if it exists
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Test SageMaker endpoint with streaming response"
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default=os.getenv("ENDPOINT_NAME"),
        help="SageMaker endpoint name (default: from .env)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION"),
        help="AWS region (default: from .env or default region)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    
    args = parser.parse_args()
    
    if not args.endpoint_name:
        parser.error(
            "Endpoint name is required. Provide via --endpoint-name or set ENDPOINT_NAME in .env"
        )
    
    test_sagemaker_endpoint(
        endpoint_name=args.endpoint_name,
        region=args.region,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
