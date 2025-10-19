#!/usr/bin/env python3
"""
Auto Scaling Test Scenarios
Tests auto scaling behavior under different load conditions
"""
import argparse
import asyncio
import json
import os
import time
import boto3
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from dotenv import load_dotenv

# Import the metrics checker from parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from check_metrics import SageMakerMetricsChecker


class AutoScalingTester:
    """Test auto scaling behavior"""
    
    def __init__(self, endpoint_name: str, region: str = None):
        """
        Initialize tester
        
        Args:
            endpoint_name: SageMaker endpoint name
            region: AWS region
        """
        self.endpoint_name = endpoint_name
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        
        self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=self.region)
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=self.region)
        
        # Get variant name
        response = self.sagemaker_client.describe_endpoint(EndpointName=self.endpoint_name)
        self.variant_name = response['ProductionVariants'][0]['VariantName']
        
        # Initialize metrics checker
        self.metrics_checker = SageMakerMetricsChecker(endpoint_name, region)
        
        # Verify endpoint status
        self._verify_endpoint_status()
    
    def _verify_endpoint_status(self):
        """Verify endpoint is ready for testing"""
        try:
            response = self.sagemaker_client.describe_endpoint(EndpointName=self.endpoint_name)
            status = response['EndpointStatus']
            
            print(f"🔍 Endpoint Status Check:")
            print(f"   Endpoint: {self.endpoint_name}")
            print(f"   Status: {status}")
            
            if status != 'InService':
                print(f"⚠️ Warning: Endpoint is not in 'InService' status!")
                print(f"   Current status: {status}")
                print(f"   Testing may fail or hang.")
                
                if status == 'Creating':
                    print(f"   Please wait for endpoint to finish creating.")
                elif status == 'Failed':
                    print(f"   Endpoint creation failed. Check CloudWatch logs.")
                
                # Ask user if they want to continue
                response = input("Continue anyway? (y/N): ")
                if response.lower() != 'y':
                    print("Exiting...")
                    exit(1)
            else:
                print(f"✅ Endpoint is ready for testing")
                
        except Exception as e:
            print(f"❌ Error checking endpoint status: {e}")
            exit(1)
    
    def get_current_instance_count(self) -> int:
        """Get current number of instances"""
        response = self.sagemaker_client.describe_endpoint(EndpointName=self.endpoint_name)
        return response['ProductionVariants'][0]['CurrentInstanceCount']
    
    def get_desired_instance_count(self) -> int:
        """Get desired number of instances"""
        response = self.sagemaker_client.describe_endpoint(EndpointName=self.endpoint_name)
        return response['ProductionVariants'][0]['DesiredInstanceCount']
    
    async def send_request(self, prompt: str, max_tokens: int = 100, timeout: int = 60) -> Dict:
        """Send a single inference request with timeout"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "do_sample": True,
                "temperature": 0.7
            }
        }
        
        start_time = time.time()
        try:
            # Add timeout to prevent hanging
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.sagemaker_runtime.invoke_endpoint,
                    EndpointName=self.endpoint_name,
                    ContentType='application/json',
                    Body=json.dumps(payload)
                ),
                timeout=timeout
            )
            
            result = json.loads(response['Body'].read().decode())
            latency = time.time() - start_time
            
            return {
                'success': True,
                'latency': latency,
                'result': result
            }
        except asyncio.TimeoutError:
            latency = time.time() - start_time
            return {
                'success': False,
                'latency': latency,
                'error': f'Request timeout after {timeout} seconds'
            }
        except Exception as e:
            latency = time.time() - start_time
            return {
                'success': False,
                'latency': latency,
                'error': str(e)
            }
    
    async def send_concurrent_requests(
        self,
        num_requests: int,
        prompt: str = "What is machine learning?",
        max_tokens: int = 100,
        timeout: int = 60
    ) -> List[Dict]:
        """Send multiple concurrent requests with timeout"""
        print(f"  Sending {num_requests} concurrent requests (timeout: {timeout}s)...")
        
        tasks = [
            self.send_request(prompt, max_tokens, timeout)
            for _ in range(num_requests)
        ]
        
        try:
            # Add overall timeout for the entire batch
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout + 30  # Extra buffer for batch timeout
            )
            
            # Convert exceptions to error results
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({
                        'success': False,
                        'latency': timeout,
                        'error': str(result)
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except asyncio.TimeoutError:
            print(f"  ⚠️ Batch timeout after {timeout + 30} seconds")
            return [{
                'success': False,
                'latency': timeout + 30,
                'error': 'Batch timeout'
            } for _ in range(num_requests)]
    
    def print_metrics_summary(self, minutes: int = 2):
        """Print concise metrics summary using the new metrics checker"""
        try:
            metrics_data = self.metrics_checker.collect_metrics(minutes=minutes)
            
            # Get key metrics
            invocations = metrics_data['invocation_metrics'].get('Invocations', {})
            concurrent = metrics_data['invocation_metrics'].get('ConcurrentRequestsPerModel', {})
            cpu = metrics_data['instance_metrics'].get('CPUUtilization', {})
            
            # Extract values
            total_requests = invocations.get('total_requests', 0) if invocations.get('type') == 'summary' else 0
            requests_per_min = invocations.get('requests_per_minute', 0) if invocations.get('type') == 'summary' else 0
            max_concurrent = concurrent.get('max_concurrent', 0) if concurrent.get('type') == 'summary' else 0
            
            # Get latest CPU from intervals
            cpu_avg = 0
            if cpu.get('type') == 'instance' and cpu.get('intervals'):
                latest_interval = cpu['intervals'][-1]  # Get most recent interval
                if latest_interval.get('has_data'):
                    cpu_avg = latest_interval.get('avg_utilization', 0)
            
            print(f"📊 Metrics ({minutes}min): Requests={total_requests:.0f} ({requests_per_min:.1f}/min), Concurrent={max_concurrent:.0f}, CPU={cpu_avg:.1f}%")
            
        except Exception as e:
            print(f"⚠️ Error getting metrics summary: {e}")
    
    def print_status(self, phase: str):
        """Print current endpoint status with metrics"""
        current_count = self.get_current_instance_count()
        desired_count = self.get_desired_instance_count()
        
        print(f"\n{'=' * 80}")
        print(f"Status - {phase}")
        print(f"{'=' * 80}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current Instances: {current_count}")
        print(f"Desired Instances: {desired_count}")
        
        # Show scaling status
        if current_count != desired_count:
            print(f"🔄 Scaling in progress: {current_count} → {desired_count}")
        elif current_count > 1:
            print(f"📈 Scaled out: {current_count} instances")
        else:
            print(f"📊 Normal: {current_count} instance")
        
        # Print metrics summary
        self.print_metrics_summary(minutes=3)
        print(f"{'=' * 80}\n")
    
    async def test_scale_out(
        self,
        num_requests: int = 100,
        duration_seconds: int = 300,
        check_interval: int = 30,
        metrics_interval: int = 30
    ):
        """
        Test scale out behavior
        
        Args:
            num_requests: Number of concurrent requests per batch
            duration_seconds: Duration to maintain load
            check_interval: Interval to check status (seconds)
            metrics_interval: Interval to print metrics summary (seconds)
        """
        print("\n" + "=" * 80)
        print("SCALE OUT TEST")
        print("=" * 80)
        print(f"Test Duration: {duration_seconds} seconds")
        print(f"Concurrent Requests per Batch: {num_requests}")
        print(f"Check Interval: {check_interval} seconds")
        print("=" * 80)
        
        initial_count = self.get_current_instance_count()
        print(f"\nInitial Instance Count: {initial_count}")
        
        # Wait a moment for any pending metrics to settle
        print("Checking initial status...")
        await asyncio.sleep(5)
        self.print_status("Before Load")
        
        # Send a warmup request to check if endpoint is responsive
        print("\nSending warmup request to verify endpoint responsiveness...")
        warmup_result = await self.send_request("Hello", max_tokens=10, timeout=120)
        
        if warmup_result['success']:
            print(f"✅ Warmup successful (latency: {warmup_result['latency']:.2f}s)")
        else:
            print(f"❌ Warmup failed: {warmup_result['error']}")
            print("⚠️ Endpoint may not be responsive. Continuing anyway...")
            await asyncio.sleep(5)
        
        print(f"\nStarting high load for {duration_seconds} seconds...")
        print("Sending concurrent requests...")
        
        start_time = time.time()
        request_count = 0
        last_metrics_time = 0
        
        while time.time() - start_time < duration_seconds:
            # Send batch of concurrent requests with timeout
            results = await self.send_concurrent_requests(num_requests, timeout=90)
            request_count += len(results)
            
            successful = sum(1 for r in results if r['success'])
            failed = len(results) - successful
            avg_latency = sum(r['latency'] for r in results) / len(results)
            
            elapsed = time.time() - start_time
            print(f"\n[{elapsed:.0f}s] Batch complete:")
            print(f"  Requests: {len(results)} (Success: {successful}, Failed: {failed})")
            print(f"  Avg Latency: {avg_latency:.2f}s")
            
            # Print metrics summary at specified interval
            if elapsed - last_metrics_time >= metrics_interval:
                print(f"  ", end="")  # Indent for alignment
                self.print_metrics_summary(minutes=2)
                last_metrics_time = elapsed
            
            # Check status periodically
            if int(elapsed) % check_interval == 0:
                # Wait a bit for metrics to update before checking
                await asyncio.sleep(5)
                self.print_status(f"During Load ({elapsed:.0f}s)")
            
            # Small delay between batches
            await asyncio.sleep(3)
        
        print(f"\n✅ Load test complete. Total requests sent: {request_count}")
        
        # Wait for scaling to complete
        print("\nWaiting for scale out to complete (checking every 30s)...")
        max_wait = 600  # 10 minutes
        wait_start = time.time()
        check_count = 0
        
        while time.time() - wait_start < max_wait:
            check_count += 1
            elapsed_wait = time.time() - wait_start
            
            print(f"\n⏳ Waiting for scale out... (Check #{check_count}, {elapsed_wait:.0f}s elapsed)")
            
            current_count = self.get_current_instance_count()
            desired_count = self.get_desired_instance_count()
            
            self.print_status("Scaling")
            
            if current_count > initial_count:
                print(f"✅ Scale out detected! Instances: {initial_count} → {current_count}")
                break
            
            if current_count == desired_count and desired_count > initial_count:
                print(f"✅ Scale out complete! Instances: {initial_count} → {current_count}")
                break
            
            print(f"💭 Still waiting... Current: {current_count}, Desired: {desired_count}, Initial: {initial_count}")
            print(f"   Next check in 30 seconds...")
            await asyncio.sleep(30)
        
        final_count = self.get_current_instance_count()
        
        print("\n" + "=" * 80)
        print("SCALE OUT TEST RESULTS")
        print("=" * 80)
        print(f"Initial Instances: {initial_count}")
        print(f"Final Instances: {final_count}")
        print(f"Scaled Out: {final_count > initial_count}")
        print(f"Total Requests: {request_count}")
        print("=" * 80)
    
    async def test_scale_in(
        self,
        wait_duration: int = 600,
        check_interval: int = 30,
        metrics_interval: int = 30
    ):
        """
        Test scale in behavior
        
        Args:
            wait_duration: Duration to wait for scale in
            check_interval: Interval to check status (seconds)
            metrics_interval: Interval to print metrics summary (seconds)
        """
        print("\n" + "=" * 80)
        print("SCALE IN TEST")
        print("=" * 80)
        print(f"Wait Duration: {wait_duration} seconds")
        print(f"Check Interval: {check_interval} seconds")
        print("=" * 80)
        
        initial_count = self.get_current_instance_count()
        print(f"\nInitial Instance Count: {initial_count}")
        
        self.print_status("Before Scale In")
        
        print(f"\nWaiting for scale in (no load)...")
        print("Checking status every 30 seconds...")
        
        start_time = time.time()
        last_metrics_time = 0
        check_count = 0
        
        while time.time() - start_time < wait_duration:
            check_count += 1
            elapsed = time.time() - start_time
            
            print(f"\n⏳ Waiting for scale in... (Check #{check_count}, {elapsed:.0f}s elapsed)")
            
            current_count = self.get_current_instance_count()
            desired_count = self.get_desired_instance_count()
            
            # Print metrics summary at specified interval
            if elapsed - last_metrics_time >= metrics_interval:
                print(f"[{elapsed:.0f}s] ", end="")
                self.print_metrics_summary(minutes=2)
                last_metrics_time = elapsed
            
            self.print_status(f"Waiting ({elapsed:.0f}s)")
            
            if current_count < initial_count:
                print(f"✅ Scale in detected! Instances: {initial_count} → {current_count}")
                break
            
            print(f"💭 Still waiting for scale in... Current: {current_count}, Initial: {initial_count}")
            print(f"   Scale in typically takes 5-10 minutes due to cooldown period...")
            print(f"   Next check in {check_interval} seconds...")
            await asyncio.sleep(check_interval)
        
        final_count = self.get_current_instance_count()
        
        print("\n" + "=" * 80)
        print("SCALE IN TEST RESULTS")
        print("=" * 80)
        print(f"Initial Instances: {initial_count}")
        print(f"Final Instances: {final_count}")
        print(f"Scaled In: {final_count < initial_count}")
        print("=" * 80)
    
    async def test_full_cycle(
        self,
        scale_out_requests: int = 100,
        scale_out_duration: int = 300,
        scale_in_wait: int = 600,
        metrics_interval: int = 30
    ):
        """
        Test full auto scaling cycle (scale out + scale in)
        
        Args:
            scale_out_requests: Number of concurrent requests for scale out
            scale_out_duration: Duration of high load
            scale_in_wait: Duration to wait for scale in
            metrics_interval: Interval to print metrics summary (seconds)
        """
        print("\n" + "=" * 80)
        print("FULL AUTO SCALING CYCLE TEST")
        print("=" * 80)
        print("This test will:")
        print("1. Apply high load to trigger scale out")
        print("2. Wait for scale out to complete")
        print("3. Stop load and wait for scale in")
        print("=" * 80)
        
        # Phase 1: Scale Out
        await self.test_scale_out(
            num_requests=scale_out_requests,
            duration_seconds=scale_out_duration,
            metrics_interval=metrics_interval
        )
        
        # Phase 2: Scale In
        print("\n\nPhase 2: Testing Scale In...")
        print("Waiting for cooldown period before scale in...")
        await asyncio.sleep(60)  # Wait 1 minute before checking scale in
        
        await self.test_scale_in(
            wait_duration=scale_in_wait,
            metrics_interval=metrics_interval
        )
        
        print("\n" + "=" * 80)
        print("✅ FULL CYCLE TEST COMPLETE")
        print("=" * 80)


async def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Test Auto Scaling behavior for SageMaker Endpoints"
    )
    
    parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="SageMaker endpoint name"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=['scale-out', 'scale-in', 'full-cycle'],
        default='full-cycle',
        help="Test scenario to run (default: full-cycle)"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of concurrent requests per batch for scale out test (default: 100)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Duration of high load in seconds for scale out test (default: 300)"
    )
    parser.add_argument(
        "--scale-in-wait",
        type=int,
        default=600,
        help="Duration to wait for scale in in seconds (default: 600)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION"),
        help="AWS region"
    )
    parser.add_argument(
        "--metrics-interval",
        type=int,
        default=30,
        help="Interval to print metrics summary in seconds (default: 30)"
    )
    
    args = parser.parse_args()
    
    tester = AutoScalingTester(
        endpoint_name=args.endpoint_name,
        region=args.region
    )
    
    if args.test == 'scale-out':
        await tester.test_scale_out(
            num_requests=args.num_requests,
            duration_seconds=args.duration,
            metrics_interval=args.metrics_interval
        )
    elif args.test == 'scale-in':
        await tester.test_scale_in(
            wait_duration=args.scale_in_wait,
            metrics_interval=args.metrics_interval
        )
    elif args.test == 'full-cycle':
        await tester.test_full_cycle(
            scale_out_requests=args.num_requests,
            scale_out_duration=args.duration,
            scale_in_wait=args.scale_in_wait,
            metrics_interval=args.metrics_interval
        )


if __name__ == "__main__":
    asyncio.run(main())
