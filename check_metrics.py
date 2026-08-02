#!/usr/bin/env python3
"""
CloudWatch Metrics Checker
Debug tool to check if metrics are being published
"""
import argparse
import boto3
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv


class SageMakerMetricsChecker:
    """SageMaker CloudWatch metrics checker and reporter"""
    
    def __init__(self, endpoint_name: str, region: str = None):
        """Initialize the metrics checker"""
        self.endpoint_name = endpoint_name
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=self.region)
        
        # Get endpoint info
        response = self.sagemaker_client.describe_endpoint(EndpointName=self.endpoint_name)
        self.variant_name = response['ProductionVariants'][0]['VariantName']
        self.endpoint_info = {
            'status': response['EndpointStatus'],
            'current_instances': response['ProductionVariants'][0]['CurrentInstanceCount'],
            'desired_instances': response['ProductionVariants'][0]['DesiredInstanceCount'],
            'variant_name': self.variant_name
        }
        
        # Metric definitions
        self.invocation_metrics = [
            'Invocations',
            'InvocationsPerInstance', 
            'ModelLatency',
            'OverheadLatency',
            'ConcurrentRequestsPerModel',
            'ConcurrentRequestsPerCopy',
            'Invocation4XXErrors',
            'Invocation5XXErrors',
            # Streaming metrics: FirstChunk* is the SageMaker-side counterpart of
            # the client-side TTFT the benchmark reports. Comparing
            # FirstChunkModelLatency (in-container) with FirstChunkLatency
            # (incl. platform routing) is the only way to attribute a slow TTFT
            # to the model versus SageMaker overhead. All in Microseconds.
            'FirstChunkLatency',
            'FirstChunkModelLatency',
            'FirstChunkOverheadLatency',
            # Failures that happen after the 200 response header - a state that
            # has no equivalent in a plain HTTP benchmark.
            'MidStreamErrors',
            'InvocationModelErrors'
        ]
        
        self.instance_metrics = [
            'CPUUtilization',
            'MemoryUtilization',
            'GPUUtilization',
            'GPUMemoryUtilization',
            'DiskUtilization'
        ]
    
    def get_metric_statistics(self, metric_name: str, start_time: datetime, end_time: datetime, namespace: str) -> List[Dict]:
        """Get CloudWatch metric statistics"""
        # Define valid statistics per metric
        if metric_name in ['Invocations']:
            valid_stats = ['Sum', 'SampleCount']
        elif metric_name in ['InvocationsPerInstance']:
            valid_stats = ['Sum']
        elif metric_name in ['ModelLatency', 'OverheadLatency',
                             'FirstChunkLatency', 'FirstChunkModelLatency',
                             'FirstChunkOverheadLatency']:
            valid_stats = ['Average', 'Sum', 'Minimum', 'Maximum', 'SampleCount']
        elif metric_name in ['ConcurrentRequestsPerModel', 'ConcurrentRequestsPerCopy']:
            # AWS documents only Min/Max as valid statistics for these two.
            valid_stats = ['Minimum', 'Maximum', 'SampleCount']
        elif metric_name in ['Invocation4XXErrors', 'Invocation5XXErrors',
                             'MidStreamErrors', 'InvocationModelErrors']:
            valid_stats = ['Average', 'Sum']
        else:
            valid_stats = ['Average', 'Maximum', 'Sum', 'SampleCount']
        
        try:
            # Try with basic dimensions first
            response = self.cloudwatch_client.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=[
                    {'Name': 'EndpointName', 'Value': self.endpoint_name},
                    {'Name': 'VariantName', 'Value': self.variant_name}
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=60,
                Statistics=valid_stats
            )
            
            # If no data found, try with EndpointConfigName dimension as well
            if not response['Datapoints'] and namespace == 'AWS/SageMaker':
                endpoint_config_name = self.sagemaker_client.describe_endpoint(EndpointName=self.endpoint_name)['EndpointConfigName']
                response = self.cloudwatch_client.get_metric_statistics(
                    Namespace=namespace,
                    MetricName=metric_name,
                    Dimensions=[
                        {'Name': 'EndpointName', 'Value': self.endpoint_name},
                        {'Name': 'VariantName', 'Value': self.variant_name},
                        {'Name': 'EndpointConfigName', 'Value': endpoint_config_name}
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=60,
                    Statistics=valid_stats
                )
            
            return sorted(response['Datapoints'], key=lambda x: x['Timestamp'], reverse=True)
        
        except Exception as e:
            print(f"⚠️ Error getting {metric_name}: {e}")
            return []
    
    def calculate_interval_aggregations(self, datapoints: List[Dict], unified_timeline: List[datetime], 
                                      start_time: datetime, metric_type: str) -> List[Dict]:
        """Calculate aggregations for time intervals"""
        results = []
        
        for i, time_point in enumerate(unified_timeline):
            # Define the interval
            if i == 0:
                interval_start = start_time
            else:
                interval_start = unified_timeline[i-1]
            interval_end = time_point
            
            # Find datapoints within this interval
            interval_datapoints = [
                dp for dp in datapoints 
                if interval_start <= dp['Timestamp'] <= interval_end
            ]
            
            if interval_datapoints:
                if metric_type == 'latency':
                    # For latency metrics, calculate weighted averages
                    interval_requests = sum(dp.get('SampleCount', 0) for dp in interval_datapoints)
                    if interval_requests > 0:
                        total_latency_time = sum(dp.get('Average', 0) * dp.get('SampleCount', 0) for dp in interval_datapoints)
                        avg_latency = total_latency_time / interval_requests
                        max_latency = max(dp.get('Maximum', 0) for dp in interval_datapoints)
                        
                        results.append({
                            'timestamp': time_point,
                            'avg_latency': avg_latency,
                            'max_latency': max_latency,
                            'requests': interval_requests,
                            'has_data': True
                        })
                    else:
                        results.append({
                            'timestamp': time_point,
                            'has_data': False
                        })
                elif metric_type == 'instance':
                    # For instance metrics, calculate simple averages
                    avg_utilization = sum(dp.get('Average', 0) for dp in interval_datapoints) / len(interval_datapoints)
                    max_utilization = max(dp.get('Maximum', 0) for dp in interval_datapoints)
                    
                    results.append({
                        'timestamp': time_point,
                        'avg_utilization': avg_utilization,
                        'max_utilization': max_utilization,
                        'datapoint_count': len(interval_datapoints),
                        'has_data': True
                    })
                else:
                    results.append({
                        'timestamp': time_point,
                        'has_data': False
                    })
            else:
                results.append({
                    'timestamp': time_point,
                    'has_data': False
                })
        
        return results
    
    def collect_metrics(self, minutes: int = 30) -> Dict:
        """Collect all metrics data"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(minutes=minutes)
        
        # Create unified timeline
        # Calculate interval size: divide total time into 5 equal intervals
        interval_minutes = minutes / 5
        time_interval = timedelta(minutes=interval_minutes)
        unified_timeline = []
        for i in range(5):
            time_point = end_time - (i * time_interval)
            unified_timeline.append(time_point)
        unified_timeline.reverse()
        
        metrics_data = {
            'endpoint_info': self.endpoint_info,
            'time_range': {
                'start': start_time,
                'end': end_time,
                'minutes': minutes,
                'interval_minutes': interval_minutes
            },
            'unified_timeline': unified_timeline,
            'invocation_metrics': {},
            'instance_metrics': {}
        }
        
        # Collect invocation metrics
        for metric_name in self.invocation_metrics:
            datapoints = self.get_metric_statistics(metric_name, start_time, end_time, 'AWS/SageMaker')
            
            if datapoints:
                if metric_name in ['ModelLatency', 'OverheadLatency',
                                   'FirstChunkLatency', 'FirstChunkModelLatency',
                                   'FirstChunkOverheadLatency']:
                    # Calculate interval aggregations for latency metrics
                    intervals = self.calculate_interval_aggregations(datapoints, unified_timeline, start_time, 'latency')
                    total_requests = sum(dp.get('SampleCount', 0) for dp in datapoints)
                    active_periods = len([dp for dp in datapoints if dp.get('SampleCount', 0) > 0])
                    
                    metrics_data['invocation_metrics'][metric_name] = {
                        'type': 'latency',
                        'total_requests': total_requests,
                        'active_periods': active_periods,
                        'intervals': intervals,
                        'raw_datapoints': len(datapoints)
                    }
                elif metric_name in ['Invocations', 'InvocationsPerInstance', 'ConcurrentRequestsPerModel']:
                    # Calculate summary statistics
                    total_requests = sum(dp.get('Sum', 0) for dp in datapoints)
                    requests_per_minute = total_requests / minutes if minutes > 0 else 0
                    
                    if metric_name == 'ConcurrentRequestsPerModel':
                        # AWS defines only Min/Max for this metric, so there is no
                        # Average field to report - use Min as the floor instead of
                        # averaging a statistic that does not exist.
                        max_concurrent = max(dp.get('Maximum', 0) for dp in datapoints)
                        min_concurrent = min(dp.get('Minimum', 0) for dp in datapoints)

                        metrics_data['invocation_metrics'][metric_name] = {
                            'type': 'summary',
                            'max_concurrent': max_concurrent,
                            'min_concurrent': min_concurrent
                        }
                    else:
                        metrics_data['invocation_metrics'][metric_name] = {
                            'type': 'summary',
                            'total_requests': total_requests,
                            'requests_per_minute': requests_per_minute,
                            'time_period': minutes
                        }
                elif metric_name in ['Invocation4XXErrors', 'Invocation5XXErrors']:
                    # Calculate error summary
                    total_errors = sum(dp.get('Sum', 0) for dp in datapoints)
                    
                    metrics_data['invocation_metrics'][metric_name] = {
                        'type': 'error_summary',
                        'total_errors': total_errors,
                        'time_period': minutes
                    }
            else:
                metrics_data['invocation_metrics'][metric_name] = {
                    'type': 'no_data'
                }
        
        # Collect instance metrics
        for metric_name in self.instance_metrics:
            datapoints = self.get_metric_statistics(metric_name, start_time, end_time, '/aws/sagemaker/Endpoints')
            
            if datapoints:
                intervals = self.calculate_interval_aggregations(datapoints, unified_timeline, start_time, 'instance')
                
                metrics_data['instance_metrics'][metric_name] = {
                    'type': 'instance',
                    'intervals': intervals,
                    'raw_datapoints': len(datapoints)
                }
            else:
                metrics_data['instance_metrics'][metric_name] = {
                    'type': 'no_data'
                }
        
        return metrics_data
    
    def print_console_report(self, metrics_data: Dict):
        """Print metrics report to console"""
        endpoint_info = metrics_data['endpoint_info']
        time_range = metrics_data['time_range']
        unified_timeline = metrics_data['unified_timeline']
        
        print("=" * 80)
        print("Checking Endpoint Configuration")
        print("=" * 80)
        print(f"Endpoint: {self.endpoint_name}")
        print(f"Variant: {endpoint_info['variant_name']}")
        print(f"Status: {endpoint_info['status']}")
        print(f"Current Instances: {endpoint_info['current_instances']}")
        print(f"Desired Instances: {endpoint_info['desired_instances']}")
        print()
        
        print("=" * 80)
        print(f"CloudWatch Metrics (Last {time_range['minutes']} minutes)")
        print("=" * 80)
        print(f"Time Range: {time_range['start'].strftime('%Y-%m-%d %H:%M:%S')} - {time_range['end'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
        timeline_str = " → ".join([t.strftime('%H:%M') for t in unified_timeline])
        print(f"Unified Timeline: {timeline_str}")
        print()
        
        # Print invocation metrics
        print("Invocation Metrics (Namespace: AWS/SageMaker):")
        print("-" * 80)
        
        for metric_name, data in metrics_data['invocation_metrics'].items():
            if data['type'] == 'no_data':
                print(f"❌ {metric_name}: No data")
                print(f"   → This could mean no requests were made or metric hasn't been published yet")
            elif data['type'] == 'summary':
                print(f"✅ {metric_name}:")
                if 'total_requests' in data:
                    print(f"   📊 Total Requests: {data['total_requests']:.0f} over {data['time_period']} minutes")
                    print(f"   📈 Average: {data['requests_per_minute']:.2f} requests/minute")
                elif 'max_concurrent' in data:
                    print(f"   📊 Max Concurrent: {data['max_concurrent']:.0f} requests")
                    print(f"   📉 Min Concurrent: {data['min_concurrent']:.0f} requests")
            elif data['type'] == 'error_summary':
                print(f"✅ {metric_name}:")
                if data['total_errors'] > 0:
                    print(f"   ❌ Total Errors: {data['total_errors']:.0f} over {data['time_period']} minutes")
                    print(f"      → Check CloudWatch logs: /aws/sagemaker/Endpoints/{self.endpoint_name}")
                else:
                    print(f"   ✅ No errors in the last {data['time_period']} minutes")
            elif data['type'] == 'latency':
                print(f"✅ {metric_name}:")
                print(f"   Summary: {data['total_requests']:.0f} total requests across {data['active_periods']} active periods")
                interval_min = metrics_data['time_range']['interval_minutes']
                print(f"   Timeline ({interval_min:.1f}-minute intervals):")
                
                for interval in data['intervals']:
                    timestamp = interval['timestamp'].strftime('%H:%M:%S')
                    if interval['has_data']:
                        # CloudWatch ModelLatency/OverheadLatency are Microseconds,
                        # so /1000 is milliseconds. The seconds figure is only a
                        # convenience restatement of the same number.
                        avg_ms = interval['avg_latency'] / 1000
                        max_ms = interval['max_latency'] / 1000
                        print(f"   [{timestamp}] Avg={avg_ms:.1f}ms (= {avg_ms / 1000:.2f}s), Max={max_ms:.1f}ms (= {max_ms / 1000:.2f}s), Requests={interval['requests']:.0f}")
                    else:
                        print(f"   [{timestamp}] No activity")
            
            print()
        
        # Print instance metrics
        print("Instance Metrics (Namespace: /aws/sagemaker/Endpoints):")
        print("-" * 80)
        
        for metric_name, data in metrics_data['instance_metrics'].items():
            if data['type'] == 'no_data':
                print(f"❌ {metric_name}: No data")
                print(f"   → Instance metrics should always be available if endpoint is running")
            elif data['type'] == 'instance':
                print(f"✅ {metric_name}:")
                interval_min = metrics_data['time_range']['interval_minutes']
                print(f"   Timeline ({interval_min:.1f}-minute intervals from {data['raw_datapoints']} datapoints):")
                
                for interval in data['intervals']:
                    timestamp = interval['timestamp'].strftime('%H:%M:%S')
                    if interval['has_data']:
                        print(f"   [{timestamp}] Avg={interval['avg_utilization']:.2f}%, Max={interval['max_utilization']:.2f}% ({interval['datapoint_count']} datapoints)")
                    else:
                        print(f"   [{timestamp}] No data")
            
            print()
        
        # Print recommendations
        print("=" * 80)
        print("Metric Interpretation & Recommendations")
        print("=" * 80)
        print("📊 Understanding the Results:")
        print("   • Invocations = 0.00: No requests sent to endpoint yet")
        print("   • ModelLatency > 0: Good! Shows endpoint processed requests")
        print("   • GPUMemoryUtilization ~91%: Model loaded in GPU memory")
        print("   • CPUUtilization low: Normal for GPU-based inference")
        print("   • 4XX/5XX Errors = 0: No client/server errors (good!)")
        print()
        print("🔗 Troubleshooting Links:")
        print("   • CloudWatch Logs: https://console.aws.amazon.com/cloudwatch/home#logsV2:log-groups")
        print("   • SageMaker Troubleshooting: https://docs.aws.amazon.com/sagemaker/latest/dg/inference-troubleshooting.html")
        print()
        print("🔧 Next Steps:")
        print("1. Send a test request to generate invocation metrics:")
        print(f"   python test_endpoint.py --endpoint-name {self.endpoint_name}")
        print()
        print("2. Wait 2-3 minutes for new metrics to appear")
        print()
        print("3. Run this script again to verify metrics are being published")
        print()
        print("4. If metrics still don't appear, check:")
        print("   - Endpoint is in 'InService' status")
        print("   - Requests are successfully processed (no 4xx/5xx errors)")
        print("   - CloudWatch permissions are configured")
        print("   - Try extending the time range (--minutes parameter)")
        print("=" * 80)
    
    def generate_markdown_report(self, metrics_data: Dict, language: str = 'en') -> str:
        """Generate Markdown report"""
        endpoint_info = metrics_data['endpoint_info']
        time_range = metrics_data['time_range']
        unified_timeline = metrics_data['unified_timeline']
        
        # Language-specific text
        if language == 'ko':
            texts = {
                'title': f"# SageMaker 엔드포인트 메트릭 리포트",
                'endpoint_config': "## 엔드포인트 구성",
                'metrics_summary': f"## CloudWatch 메트릭 (최근 {time_range['minutes']}분)",
                'invocation_metrics': "### 호출 메트릭 (AWS/SageMaker)",
                'instance_metrics': "### 인스턴스 메트릭 (/aws/sagemaker/Endpoints)",
                'recommendations': "## 메트릭 해석 및 권장사항",
                'understanding': "### 결과 이해하기",
                'troubleshooting': "### 문제 해결 링크",
                'next_steps': "### 다음 단계",
                'endpoint': "엔드포인트",
                'variant': "변형",
                'status': "상태",
                'current_instances': "현재 인스턴스",
                'desired_instances': "목표 인스턴스",
                'time_range': "시간 범위",
                'unified_timeline': "통합 타임라인",
                'total_requests': "총 요청",
                'average': "평균",
                'requests_per_minute': "요청/분",
                'max_concurrent': "최대 동시",
                'min_concurrent': "최소 동시",
                'no_errors': "오류 없음",
                'total_errors': "총 오류",
                'summary': "요약",
                'timeline': "타임라인",
                'no_activity': "활동 없음",
                'no_data': "데이터 없음",
                'datapoints': "데이터포인트"
            }
        else:  # English
            texts = {
                'title': f"# SageMaker Endpoint Metrics Report",
                'endpoint_config': "## Endpoint Configuration",
                'metrics_summary': f"## CloudWatch Metrics (Last {time_range['minutes']} minutes)",
                'invocation_metrics': "### Invocation Metrics (AWS/SageMaker)",
                'instance_metrics': "### Instance Metrics (/aws/sagemaker/Endpoints)",
                'recommendations': "## Metric Interpretation & Recommendations",
                'understanding': "### Understanding the Results",
                'troubleshooting': "### Troubleshooting Links",
                'next_steps': "### Next Steps",
                'endpoint': "Endpoint",
                'variant': "Variant",
                'status': "Status",
                'current_instances': "Current Instances",
                'desired_instances': "Desired Instances",
                'time_range': "Time Range",
                'unified_timeline': "Unified Timeline",
                'total_requests': "Total Requests",
                'average': "Average",
                'requests_per_minute': "requests/minute",
                'max_concurrent': "Max Concurrent",
                'min_concurrent': "Min Concurrent",
                'no_errors': "No errors",
                'total_errors': "Total Errors",
                'summary': "Summary",
                'timeline': "Timeline",
                'no_activity': "No activity",
                'no_data': "No data",
                'datapoints': "datapoints"
            }
        
        # Generate report
        report = []
        report.append(texts['title'])
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"**Endpoint:** `{self.endpoint_name}`\n")
        
        # Endpoint configuration
        report.append(texts['endpoint_config'])
        report.append("")
        report.append(f"- **{texts['endpoint']}:** `{self.endpoint_name}`")
        report.append(f"- **{texts['variant']}:** `{endpoint_info['variant_name']}`")
        report.append(f"- **{texts['status']}:** `{endpoint_info['status']}`")
        report.append(f"- **{texts['current_instances']}:** {endpoint_info['current_instances']}")
        report.append(f"- **{texts['desired_instances']}:** {endpoint_info['desired_instances']}")
        report.append("")
        
        # Metrics summary
        report.append(texts['metrics_summary'])
        report.append("")
        report.append(f"- **{texts['time_range']}:** {time_range['start'].strftime('%Y-%m-%d %H:%M:%S')} - {time_range['end'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
        timeline_str = " → ".join([t.strftime('%H:%M') for t in unified_timeline])
        report.append(f"- **{texts['unified_timeline']}:** {timeline_str}")
        report.append("")
        
        # Invocation metrics
        report.append(texts['invocation_metrics'])
        report.append("")
        
        for metric_name, data in metrics_data['invocation_metrics'].items():
            report.append(f"#### {metric_name}")
            report.append("")
            
            if data['type'] == 'no_data':
                report.append(f"❌ **{texts['no_data']}**")
            elif data['type'] == 'summary':
                if 'total_requests' in data:
                    report.append(f"- **{texts['total_requests']}:** {data['total_requests']:.0f} over {data['time_period']} minutes")
                    report.append(f"- **{texts['average']}:** {data['requests_per_minute']:.2f} {texts['requests_per_minute']}")
                elif 'max_concurrent' in data:
                    report.append(f"- **{texts['max_concurrent']}:** {data['max_concurrent']:.0f} requests")
                    report.append(f"- **{texts['min_concurrent']}:** {data['min_concurrent']:.0f} requests")
            elif data['type'] == 'error_summary':
                if data['total_errors'] > 0:
                    report.append(f"❌ **{texts['total_errors']}:** {data['total_errors']:.0f} over {data['time_period']} minutes")
                else:
                    report.append(f"✅ **{texts['no_errors']}** in the last {data['time_period']} minutes")
            elif data['type'] == 'latency':
                report.append(f"- **{texts['summary']}:** {data['total_requests']:.0f} total requests across {data['active_periods']} active periods")
                interval_min = time_range['interval_minutes']
                report.append(f"- **{texts['timeline']}** ({interval_min:.1f}-minute intervals):")
                report.append("")
                
                for interval in data['intervals']:
                    timestamp = interval['timestamp'].strftime('%H:%M:%S')
                    if interval['has_data']:
                        # CloudWatch ModelLatency/OverheadLatency are Microseconds.
                        avg_ms = interval['avg_latency'] / 1000
                        max_ms = interval['max_latency'] / 1000
                        report.append(f"  - `[{timestamp}]` Avg={avg_ms:.1f}ms (= {avg_ms / 1000:.2f}s), Max={max_ms:.1f}ms (= {max_ms / 1000:.2f}s), Requests={interval['requests']:.0f}")
                    else:
                        report.append(f"  - `[{timestamp}]` {texts['no_activity']}")
            
            report.append("")
        
        # Instance metrics
        report.append(texts['instance_metrics'])
        report.append("")
        
        for metric_name, data in metrics_data['instance_metrics'].items():
            report.append(f"#### {metric_name}")
            report.append("")
            
            if data['type'] == 'no_data':
                report.append(f"❌ **{texts['no_data']}**")
            elif data['type'] == 'instance':
                interval_min = time_range['interval_minutes']
                report.append(f"- **{texts['timeline']}** ({interval_min:.1f}-minute intervals from {data['raw_datapoints']} {texts['datapoints']}):")
                report.append("")
                
                for interval in data['intervals']:
                    timestamp = interval['timestamp'].strftime('%H:%M:%S')
                    if interval['has_data']:
                        report.append(f"  - `[{timestamp}]` Avg={interval['avg_utilization']:.2f}%, Max={interval['max_utilization']:.2f}% ({interval['datapoint_count']} {texts['datapoints']})")
                    else:
                        report.append(f"  - `[{timestamp}]` {texts['no_data']}")
            
            report.append("")
        
        # Recommendations
        report.append(texts['recommendations'])
        report.append("")
        
        if language == 'ko':
            report.extend([
                "### 결과 이해하기",
                "",
                "- **Invocations = 0.00:** 아직 엔드포인트로 요청이 전송되지 않음",
                "- **ModelLatency > 0:** 좋음! 엔드포인트가 요청을 처리했음을 보여줌",
                "- **GPUMemoryUtilization ~91%:** 모델이 GPU 메모리에 로드됨",
                "- **CPUUtilization 낮음:** GPU 기반 추론에서 정상",
                "- **4XX/5XX Errors = 0:** 클라이언트/서버 오류 없음 (좋음!)",
                "",
                "### 문제 해결 링크",
                "",
                "- [CloudWatch 로그](https://console.aws.amazon.com/cloudwatch/home#logsV2:log-groups)",
                "- [SageMaker 문제 해결](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-troubleshooting.html)",
                "",
                "### 다음 단계",
                "",
                "1. 테스트 요청을 보내서 호출 메트릭 생성:",
                f"   ```bash",
                f"   python test_endpoint.py --endpoint-name {self.endpoint_name}",
                f"   ```",
                "",
                "2. 새로운 메트릭이 나타날 때까지 2-3분 대기",
                "",
                "3. 이 스크립트를 다시 실행하여 메트릭 게시 확인",
                "",
                "4. 메트릭이 여전히 나타나지 않으면 다음을 확인:",
                "   - 엔드포인트가 'InService' 상태인지",
                "   - 요청이 성공적으로 처리되는지 (4xx/5xx 오류 없음)",
                "   - CloudWatch 권한이 구성되어 있는지",
                "   - 시간 범위 확장 시도 (--minutes 매개변수)"
            ])
        else:
            report.extend([
                "### Understanding the Results",
                "",
                "- **Invocations = 0.00:** No requests sent to endpoint yet",
                "- **ModelLatency > 0:** Good! Shows endpoint processed requests",
                "- **GPUMemoryUtilization ~91%:** Model loaded in GPU memory",
                "- **CPUUtilization low:** Normal for GPU-based inference",
                "- **4XX/5XX Errors = 0:** No client/server errors (good!)",
                "",
                "### Troubleshooting Links",
                "",
                "- [CloudWatch Logs](https://console.aws.amazon.com/cloudwatch/home#logsV2:log-groups)",
                "- [SageMaker Troubleshooting](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-troubleshooting.html)",
                "",
                "### Next Steps",
                "",
                "1. Send a test request to generate invocation metrics:",
                f"   ```bash",
                f"   python test_endpoint.py --endpoint-name {self.endpoint_name}",
                f"   ```",
                "",
                "2. Wait 2-3 minutes for new metrics to appear",
                "",
                "3. Run this script again to verify metrics are being published",
                "",
                "4. If metrics still don't appear, check:",
                "   - Endpoint is in 'InService' status",
                "   - Requests are successfully processed (no 4xx/5xx errors)",
                "   - CloudWatch permissions are configured",
                "   - Try extending the time range (--minutes parameter)"
            ])
        
        return "\n".join(report)


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Check CloudWatch metrics for SageMaker Endpoint"
    )
    
    parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="SageMaker endpoint name"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION"),
        help="AWS region"
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=30,
        help="Time range in minutes to check metrics (default: 30)"
    )
    parser.add_argument(
        "--output-md",
        type=str,
        help="Output Markdown report to file"
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=['en', 'ko'],
        default='en',
        help="Report language (en: English, ko: Korean)"
    )
    
    args = parser.parse_args()
    
    # Initialize checker
    checker = SageMakerMetricsChecker(
        endpoint_name=args.endpoint_name,
        region=args.region
    )
    
    # Collect metrics
    metrics_data = checker.collect_metrics(minutes=args.minutes)
    
    # Print console report
    checker.print_console_report(metrics_data)
    
    # Generate Markdown report if requested
    if args.output_md:
        markdown_report = checker.generate_markdown_report(metrics_data, args.language)
        
        with open(args.output_md, 'w', encoding='utf-8') as f:
            f.write(markdown_report)
        
        print(f"\n📄 Markdown report saved to: {args.output_md}")


if __name__ == "__main__":
    main()