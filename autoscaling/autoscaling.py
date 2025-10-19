#!/usr/bin/env python3
"""
SageMaker Endpoint Auto Scaling Configuration
Configures auto scaling policies based on CloudWatch metrics
"""
import argparse
import os
import time
import boto3
from dotenv import load_dotenv
from typing import Optional, Dict, List


class SageMakerAutoScaling:
    """Manage SageMaker Endpoint Auto Scaling"""
    
    def __init__(self, endpoint_name: str, region: str = None):
        """
        Initialize Auto Scaling manager
        
        Args:
            endpoint_name: SageMaker endpoint name
            region: AWS region
        """
        self.endpoint_name = endpoint_name
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.region)
        self.autoscaling_client = boto3.client('application-autoscaling', region_name=self.region)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=self.region)
        
        # Get variant name from endpoint
        self.variant_name = self._get_variant_name()
        self.resource_id = f"endpoint/{self.endpoint_name}/variant/{self.variant_name}"
    
    def _get_variant_name(self) -> str:
        """Get the production variant name from endpoint"""
        try:
            response = self.sagemaker_client.describe_endpoint(
                EndpointName=self.endpoint_name
            )
            return response['ProductionVariants'][0]['VariantName']
        except Exception as e:
            raise ValueError(f"Failed to get variant name: {e}")
    
    def configure_autoscaling(
        self,
        min_capacity: int = 1,
        max_capacity: int = 10,
        target_invocations_per_instance: Optional[int] = None,
        target_concurrent_requests_per_model: Optional[int] = None,
        target_concurrent_requests_per_copy: Optional[int] = None,
        scale_in_cooldown: int = 300,
        scale_out_cooldown: int = 60,
    ) -> Dict:
        """
        Configure auto scaling for the endpoint
        
        Args:
            min_capacity: Minimum number of instances
            max_capacity: Maximum number of instances
            target_invocations_per_instance: Target invocations per instance (default: 1000)
            target_concurrent_requests_per_model: Target concurrent requests per model
            target_concurrent_requests_per_copy: Target concurrent requests per copy
            scale_in_cooldown: Cooldown period for scale in (seconds)
            scale_out_cooldown: Cooldown period for scale out (seconds)
        
        Returns:
            Dictionary with configuration results
        """
        results = {
            'endpoint_name': self.endpoint_name,
            'resource_id': self.resource_id,
            'min_capacity': min_capacity,
            'max_capacity': max_capacity,
            'policies': []
        }
        
        print("=" * 80)
        print("Configuring Auto Scaling")
        print("=" * 80)
        print(f"Endpoint: {self.endpoint_name}")
        print(f"Variant: {self.variant_name}")
        print(f"Resource ID: {self.resource_id}")
        print(f"Min Capacity: {min_capacity}")
        print(f"Max Capacity: {max_capacity}")
        print()
        
        # Register scalable target
        print("Registering scalable target...")
        try:
            self.autoscaling_client.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=self.resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=min_capacity,
                MaxCapacity=max_capacity
            )
            print("✅ Scalable target registered")
        except Exception as e:
            print(f"⚠️ Error registering scalable target: {e}")
            raise
        
        print()
        
        # Configure scaling policies
        policies_configured = 0
        
        # Policy 1: InvocationsPerInstance
        if target_invocations_per_instance is not None:
            policy_name = f"{self.endpoint_name}-invocations-per-instance"
            print(f"Configuring policy: {policy_name}")
            print(f"  Metric: InvocationsPerInstance")
            print(f"  Target: {target_invocations_per_instance}")
            
            try:
                response = self.autoscaling_client.put_scaling_policy(
                    PolicyName=policy_name,
                    ServiceNamespace='sagemaker',
                    ResourceId=self.resource_id,
                    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                    PolicyType='TargetTrackingScaling',
                    TargetTrackingScalingPolicyConfiguration={
                        'TargetValue': float(target_invocations_per_instance),
                        'PredefinedMetricSpecification': {
                            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                        },
                        'ScaleInCooldown': scale_in_cooldown,
                        'ScaleOutCooldown': scale_out_cooldown
                    }
                )
                print(f"✅ Policy created: {policy_name}")
                results['policies'].append({
                    'name': policy_name,
                    'metric': 'InvocationsPerInstance',
                    'target': target_invocations_per_instance,
                    'arn': response['PolicyARN']
                })
                policies_configured += 1
            except Exception as e:
                print(f"⚠️ Error creating policy: {e}")
            print()
        
        # Policy 2: ConcurrentRequestsPerModel (High-Resolution Predefined Metric)
        if target_concurrent_requests_per_model is not None:
            policy_name = f"{self.endpoint_name}-concurrent-requests-per-model"
            print(f"Configuring policy: {policy_name}")
            print(f"  Metric: ConcurrentRequestsPerModel (High-Resolution)")
            print(f"  Target: {target_concurrent_requests_per_model}")
            
            try:
                response = self.autoscaling_client.put_scaling_policy(
                    PolicyName=policy_name,
                    ServiceNamespace='sagemaker',
                    ResourceId=self.resource_id,
                    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                    PolicyType='TargetTrackingScaling',
                    TargetTrackingScalingPolicyConfiguration={
                        'TargetValue': float(target_concurrent_requests_per_model),
                        'PredefinedMetricSpecification': {
                            'PredefinedMetricType': 'SageMakerVariantConcurrentRequestsPerModelHighResolution'
                        },
                        'ScaleInCooldown': scale_in_cooldown,
                        'ScaleOutCooldown': scale_out_cooldown
                    }
                )
                print(f"✅ Policy created: {policy_name}")
                results['policies'].append({
                    'name': policy_name,
                    'metric': 'ConcurrentRequestsPerModel',
                    'target': target_concurrent_requests_per_model,
                    'arn': response['PolicyARN']
                })
                policies_configured += 1
            except Exception as e:
                print(f"⚠️ Error creating policy: {e}")
            print()
        
        # Policy 3: ConcurrentRequestsPerCopy (High-Resolution Predefined Metric)
        if target_concurrent_requests_per_copy is not None:
            policy_name = f"{self.endpoint_name}-concurrent-requests-per-copy"
            print(f"Configuring policy: {policy_name}")
            print(f"  Metric: ConcurrentRequestsPerCopy (High-Resolution)")
            print(f"  Target: {target_concurrent_requests_per_copy}")
            
            try:
                response = self.autoscaling_client.put_scaling_policy(
                    PolicyName=policy_name,
                    ServiceNamespace='sagemaker',
                    ResourceId=self.resource_id,
                    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                    PolicyType='TargetTrackingScaling',
                    TargetTrackingScalingPolicyConfiguration={
                        'TargetValue': float(target_concurrent_requests_per_copy),
                        'PredefinedMetricSpecification': {
                            'PredefinedMetricType': 'SageMakerVariantConcurrentRequestsPerCopyHighResolution'
                        },
                        'ScaleInCooldown': scale_in_cooldown,
                        'ScaleOutCooldown': scale_out_cooldown
                    }
                )
                print(f"✅ Policy created: {policy_name}")
                results['policies'].append({
                    'name': policy_name,
                    'metric': 'ConcurrentRequestsPerCopy',
                    'target': target_concurrent_requests_per_copy,
                    'arn': response['PolicyARN']
                })
                policies_configured += 1
            except Exception as e:
                print(f"⚠️ Error creating policy: {e}")
            print()
        
        if policies_configured == 0:
            print("⚠️ No policies configured. Please specify at least one target metric.")
            print("   Use --target-invocations-per-instance or other target options.")
        
        print("=" * 80)
        print(f"✅ Auto Scaling Configuration Complete ({policies_configured} policies)")
        print("=" * 80)
        
        return results
    
    def describe_autoscaling(self) -> Dict:
        """Describe current auto scaling configuration"""
        print("=" * 80)
        print("Auto Scaling Configuration")
        print("=" * 80)
        print(f"Endpoint: {self.endpoint_name}")
        print(f"Resource ID: {self.resource_id}")
        print()
        
        # Get scalable target
        try:
            response = self.autoscaling_client.describe_scalable_targets(
                ServiceNamespace='sagemaker',
                ResourceIds=[self.resource_id],
                ScalableDimension='sagemaker:variant:DesiredInstanceCount'
            )
            
            if response['ScalableTargets']:
                target = response['ScalableTargets'][0]
                print("Scalable Target:")
                print(f"  Min Capacity: {target['MinCapacity']}")
                print(f"  Max Capacity: {target['MaxCapacity']}")
                print()
            else:
                print("⚠️ No scalable target configured")
                return {}
        except Exception as e:
            print(f"⚠️ Error describing scalable target: {e}")
            return {}
        
        # Get scaling policies
        try:
            response = self.autoscaling_client.describe_scaling_policies(
                ServiceNamespace='sagemaker',
                ResourceId=self.resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount'
            )
            
            if response['ScalingPolicies']:
                print(f"Scaling Policies ({len(response['ScalingPolicies'])}):")
                for policy in response['ScalingPolicies']:
                    print(f"\n  Policy: {policy['PolicyName']}")
                    print(f"  Type: {policy['PolicyType']}")
                    
                    if 'TargetTrackingScalingPolicyConfiguration' in policy:
                        config = policy['TargetTrackingScalingPolicyConfiguration']
                        print(f"  Target Value: {config['TargetValue']}")
                        
                        if 'PredefinedMetricSpecification' in config:
                            metric = config['PredefinedMetricSpecification']['PredefinedMetricType']
                            print(f"  Metric: {metric}")
                        elif 'CustomizedMetricSpecification' in config:
                            metric_spec = config['CustomizedMetricSpecification']
                            print(f"  Metric: {metric_spec['MetricName']}")
                            print(f"  Namespace: {metric_spec['Namespace']}")
                        
                        print(f"  Scale In Cooldown: {config.get('ScaleInCooldown', 'N/A')}s")
                        print(f"  Scale Out Cooldown: {config.get('ScaleOutCooldown', 'N/A')}s")
            else:
                print("⚠️ No scaling policies configured")
        except Exception as e:
            print(f"⚠️ Error describing scaling policies: {e}")
        
        print()
        print("=" * 80)
        
        return response
    
    def delete_autoscaling(self) -> None:
        """Delete all auto scaling policies and deregister scalable target"""
        print("=" * 80)
        print("Deleting Auto Scaling Configuration")
        print("=" * 80)
        print(f"Endpoint: {self.endpoint_name}")
        print(f"Resource ID: {self.resource_id}")
        print()
        
        # Delete scaling policies
        try:
            response = self.autoscaling_client.describe_scaling_policies(
                ServiceNamespace='sagemaker',
                ResourceId=self.resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount'
            )
            
            for policy in response['ScalingPolicies']:
                policy_name = policy['PolicyName']
                print(f"Deleting policy: {policy_name}")
                try:
                    self.autoscaling_client.delete_scaling_policy(
                        PolicyName=policy_name,
                        ServiceNamespace='sagemaker',
                        ResourceId=self.resource_id,
                        ScalableDimension='sagemaker:variant:DesiredInstanceCount'
                    )
                    print(f"✅ Policy deleted: {policy_name}")
                except Exception as e:
                    print(f"⚠️ Error deleting policy: {e}")
        except Exception as e:
            print(f"⚠️ Error listing policies: {e}")
        
        print()
        
        # Deregister scalable target
        print("Deregistering scalable target...")
        try:
            self.autoscaling_client.deregister_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=self.resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount'
            )
            print("✅ Scalable target deregistered")
        except Exception as e:
            print(f"⚠️ Error deregistering scalable target: {e}")
        
        print()
        print("=" * 80)
        print("✅ Auto Scaling Configuration Deleted")
        print("=" * 80)
    
    def get_cloudwatch_metrics(self, period_minutes: int = 5) -> Dict:
        """
        Get CloudWatch metrics for the endpoint
        
        Args:
            period_minutes: Time period to query (minutes)
        
        Returns:
            Dictionary with metric data
        """
        import datetime
        
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(minutes=period_minutes)
        
        metrics = {}
        metric_names = [
            'InvocationsPerInstance',
            'ConcurrentRequestsPerModel',
            'ConcurrentRequestsPerCopy'
        ]
        
        print("=" * 80)
        print(f"CloudWatch Metrics (Last {period_minutes} minutes)")
        print("=" * 80)
        print(f"Endpoint: {self.endpoint_name}")
        print(f"Period: {start_time.strftime('%Y-%m-%d %H:%M:%S')} - {end_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print()
        
        for metric_name in metric_names:
            try:
                response = self.cloudwatch_client.get_metric_statistics(
                    Namespace='/aws/sagemaker/Endpoints',
                    MetricName=metric_name,
                    Dimensions=[
                        {'Name': 'EndpointName', 'Value': self.endpoint_name},
                        {'Name': 'VariantName', 'Value': self.variant_name}
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=60,  # 1 minute
                    Statistics=['Average', 'Maximum', 'Minimum']
                )
                
                datapoints = response['Datapoints']
                if datapoints:
                    # Sort by timestamp
                    datapoints.sort(key=lambda x: x['Timestamp'])
                    
                    avg_values = [dp['Average'] for dp in datapoints]
                    max_values = [dp['Maximum'] for dp in datapoints]
                    min_values = [dp['Minimum'] for dp in datapoints]
                    
                    metrics[metric_name] = {
                        'average': sum(avg_values) / len(avg_values),
                        'max': max(max_values),
                        'min': min(min_values),
                        'datapoints': len(datapoints)
                    }
                    
                    print(f"{metric_name}:")
                    print(f"  Average: {metrics[metric_name]['average']:.2f}")
                    print(f"  Max: {metrics[metric_name]['max']:.2f}")
                    print(f"  Min: {metrics[metric_name]['min']:.2f}")
                    print(f"  Data Points: {metrics[metric_name]['datapoints']}")
                else:
                    print(f"{metric_name}: No data available")
                    metrics[metric_name] = None
            except Exception as e:
                print(f"{metric_name}: Error - {e}")
                metrics[metric_name] = None
            
            print()
        
        print("=" * 80)
        
        return metrics


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Configure Auto Scaling for SageMaker Endpoints"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Configure command
    config_parser = subparsers.add_parser('configure', help='Configure auto scaling')
    config_parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="SageMaker endpoint name"
    )
    config_parser.add_argument(
        "--min-capacity",
        type=int,
        default=1,
        help="Minimum number of instances (default: 1)"
    )
    config_parser.add_argument(
        "--max-capacity",
        type=int,
        default=10,
        help="Maximum number of instances (default: 10)"
    )
    config_parser.add_argument(
        "--target-invocations-per-instance",
        type=int,
        help="Target invocations per instance (e.g., 1000)"
    )
    config_parser.add_argument(
        "--target-concurrent-requests-per-model",
        type=int,
        help="Target concurrent requests per model (e.g., 50)"
    )
    config_parser.add_argument(
        "--target-concurrent-requests-per-copy",
        type=int,
        help="Target concurrent requests per copy (e.g., 10)"
    )
    config_parser.add_argument(
        "--scale-in-cooldown",
        type=int,
        default=300,
        help="Scale in cooldown period in seconds (default: 300)"
    )
    config_parser.add_argument(
        "--scale-out-cooldown",
        type=int,
        default=60,
        help="Scale out cooldown period in seconds (default: 60)"
    )
    config_parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION"),
        help="AWS region"
    )
    
    # Describe command
    describe_parser = subparsers.add_parser('describe', help='Describe auto scaling configuration')
    describe_parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="SageMaker endpoint name"
    )
    describe_parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION"),
        help="AWS region"
    )
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete auto scaling configuration')
    delete_parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="SageMaker endpoint name"
    )
    delete_parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION"),
        help="AWS region"
    )
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Get CloudWatch metrics')
    metrics_parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="SageMaker endpoint name"
    )
    metrics_parser.add_argument(
        "--period",
        type=int,
        default=5,
        help="Time period in minutes (default: 5)"
    )
    metrics_parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION"),
        help="AWS region"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    autoscaling = SageMakerAutoScaling(
        endpoint_name=args.endpoint_name,
        region=args.region
    )
    
    if args.command == 'configure':
        autoscaling.configure_autoscaling(
            min_capacity=args.min_capacity,
            max_capacity=args.max_capacity,
            target_invocations_per_instance=args.target_invocations_per_instance,
            target_concurrent_requests_per_model=args.target_concurrent_requests_per_model,
            target_concurrent_requests_per_copy=args.target_concurrent_requests_per_copy,
            scale_in_cooldown=args.scale_in_cooldown,
            scale_out_cooldown=args.scale_out_cooldown
        )
    elif args.command == 'describe':
        autoscaling.describe_autoscaling()
    elif args.command == 'delete':
        autoscaling.delete_autoscaling()
    elif args.command == 'metrics':
        a


if __name__ == "__main__":
    main()
