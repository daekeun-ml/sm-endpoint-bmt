#!/usr/bin/env python3
"""
SageMaker Endpoint Creation Script
Creates a SageMaker endpoint for LLM inference using vLLM/LMI container
"""
import argparse
import json
import os
import time
import boto3
from dotenv import load_dotenv


def load_vllm_config(config_file: str) -> dict:
    """Load vLLM configuration from JSON file"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"vLLM config file not found: {config_file}\n"
            f"Please create a config file in the config/ directory.\n"
            f"Example: cp config/vllm_config.json.example {config_file}"
        )
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Remove comment fields
    config = {k: v for k, v in config.items() if not k.startswith('_')}
    
    return config


def create_endpoint(
    vllm_config_file: str,
    endpoint_name: str = None,
    instance_type: str = "ml.g5.xlarge",
    instance_count: int = 1,
    region: str = None,
    sagemaker_role: str = None,
    container_version: str = "0.34.0-lmi16.0.0-cu128",
    health_check_timeout: int = 1800,
    wait: bool = True,
):
    """
    Create a SageMaker endpoint for LLM inference
    
    Args:
        vllm_config_file: Path to vLLM configuration JSON file
        endpoint_name: Custom endpoint name (optional, auto-generated if not provided)
        instance_type: SageMaker instance type
        instance_count: Number of instances
        region: AWS region
        sagemaker_role: SageMaker execution role ARN (optional, auto-detected if not provided)
        container_version: LMI container version
        health_check_timeout: Health check timeout in seconds
        wait: Wait for endpoint creation to complete
    """
    # Import sagemaker only when needed for create
    import sagemaker
    from sagemaker import Model
    
    # Load vLLM configuration
    print("=" * 60)
    print("Loading vLLM Configuration")
    print("=" * 60)
    print(f"Config file: {vllm_config_file}")
    vllm_config = load_vllm_config(vllm_config_file)
    
    model_id = vllm_config.get("HF_MODEL_ID")
    if not model_id:
        raise ValueError("HF_MODEL_ID must be specified in vLLM config file")
    
    print(f"Model ID: {model_id}")
    print()
    
    # Initialize SageMaker session
    sess = sagemaker.session.Session(boto3.Session(region_name=region))
    region = region or sess.boto_region_name
    
    # Get SageMaker execution role
    if sagemaker_role:
        role = sagemaker_role
        print(f"Using provided role: {role}")
    else:
        try:
            role = sagemaker.get_execution_role()
            print(f"Auto-detected role: {role}")
        except ValueError:
            # If not running in SageMaker, try to get default role
            iam_client = boto3.client('iam', region_name=region)
            try:
                # Try to find a SageMaker execution role
                response = iam_client.list_roles()
                sagemaker_roles = [
                    r['Arn'] for r in response['Roles']
                    if 'SageMaker' in r['RoleName'] or 'sagemaker' in r['RoleName']
                ]
                if sagemaker_roles:
                    role = sagemaker_roles[0]
                    print(f"Found SageMaker role: {role}")
                else:
                    raise ValueError(
                        "No SageMaker execution role found. Please specify --sagemaker-role or set SAGEMAKER_ROLE in .env"
                    )
            except Exception as e:
                raise ValueError(
                    f"Could not find SageMaker execution role: {e}\n"
                    "Please specify --sagemaker-role or set SAGEMAKER_ROLE in .env"
                )
    
    print("=" * 60)
    print("SageMaker Endpoint Creation")
    print("=" * 60)
    print(f"Instance Type: {instance_type}")
    print(f"Instance Count: {instance_count}")
    print(f"Region: {region}")
    print(f"Role: {role}")
    print()
    
    # Construct container URI
    container_uri = f'763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:{container_version}'
    print(f"Container URI: {container_uri}")
    
    print("\nvLLM Environment Configuration:")
    for key, value in vllm_config.items():
        if key == "HF_TOKEN" and value:
            print(f"  {key}: ***")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Create model
    model = Model(
        image_uri=container_uri,
        role=role,
        env=vllm_config,
        sagemaker_session=sess
    )
    
    # Generate endpoint name if not provided
    if endpoint_name is None or endpoint_name == "":
        model_name = model_id.split('/')[-1]
        endpoint_name = sagemaker.utils.name_from_base(model_name)
    
    print(f"Endpoint Name: {endpoint_name}")
    print()
    
    # Deploy model
    print("Deploying model...")
    print("This may take 10-20 minutes depending on the model size.")
    print()
    
    try:
        predictor = model.deploy(
            initial_instance_count=instance_count,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            container_startup_health_check_timeout=health_check_timeout,
            wait=wait
        )
        
        if wait:
            print()
            print("=" * 60)
            print("✅ Endpoint created successfully!")
            print("=" * 60)
            print(f"Endpoint Name: {endpoint_name}")
            print(f"Region: {region}")
            print()
            print("Console URL:")
            console_url = f"https://console.aws.amazon.com/sagemaker/home?region={region}#/endpoints/{endpoint_name}"
            print(console_url)
            print()
            print("Test the endpoint:")
            print(f"  python test_endpoint.py")
            print()
            print("Run benchmark:")
            print(f"  python sagemaker_benchmark.py --endpoint-name '{endpoint_name}'")
        else:
            print()
            print("=" * 60)
            print("🚀 Endpoint creation initiated")
            print("=" * 60)
            print(f"Endpoint Name: {endpoint_name}")
            print()
            print("Check status:")
            print(f"  aws sagemaker describe-endpoint --endpoint-name {endpoint_name}")
            print()
            print("Or visit the console:")
            console_url = f"https://console.aws.amazon.com/sagemaker/home?region={region}#/endpoints/{endpoint_name}"
            print(console_url)
        
        return endpoint_name
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ Error creating endpoint")
        print("=" * 60)
        print(f"Error: {str(e)}")
        print()
        raise


def delete_endpoint(endpoint_name: str, region: str = None):
    """Delete a SageMaker endpoint and its configuration"""
    client = boto3.client('sagemaker', region_name=region)
    
    print(f"Deleting endpoint: {endpoint_name}")
    print()
    
    # Delete endpoint
    try:
        client.delete_endpoint(EndpointName=endpoint_name)
        print(f"✅ Endpoint deleted: {endpoint_name}")
    except client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            print(f"⚠️ Endpoint not found: {endpoint_name}")
        else:
            print(f"⚠️ Error deleting endpoint: {e}")
    except Exception as e:
        print(f"⚠️ Error deleting endpoint: {e}")
    
    # Delete endpoint configuration
    try:
        client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"✅ Endpoint config deleted: {endpoint_name}")
    except client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            print(f"⚠️ Endpoint config not found: {endpoint_name}")
        else:
            print(f"⚠️ Error deleting endpoint config: {e}")
    except Exception as e:
        print(f"⚠️ Error deleting endpoint config: {e}")
    
    print()
    print("Deletion complete!")


def main():
    # Load .env file if it exists
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Create or delete SageMaker endpoints for LLM inference"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create endpoint command
    create_parser = subparsers.add_parser('create', help='Create a new endpoint')
    create_parser.add_argument(
        "--vllm-config",
        type=str,
        default=os.getenv("VLLM_CONFIG_FILE", "config/vllm_config.json"),
        help="Path to vLLM configuration JSON file (default: from .env or config/vllm_config.json)"
    )
    create_parser.add_argument(
        "--endpoint-name",
        type=str,
        default=os.getenv("ENDPOINT_NAME"),
        help="Custom endpoint name (default: from .env or auto-generated)"
    )
    create_parser.add_argument(
        "--instance-type",
        type=str,
        default=os.getenv("INSTANCE_TYPE", "ml.g5.xlarge"),
        help="SageMaker instance type (default: from .env or ml.g5.xlarge)"
    )
    create_parser.add_argument(
        "--instance-count",
        type=int,
        default=int(os.getenv("INSTANCE_COUNT", "1")),
        help="Number of instances (default: from .env or 1)"
    )
    create_parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION"),
        help="AWS region (default: from .env or default region)"
    )
    create_parser.add_argument(
        "--sagemaker-role",
        type=str,
        default=os.getenv("SAGEMAKER_ROLE"),
        help="SageMaker execution role ARN (default: from .env or auto-detect)"
    )
    create_parser.add_argument(
        "--container-version",
        type=str,
        default=os.getenv("CONTAINER_VERSION", "0.34.0-lmi16.0.0-cu128"),
        help="LMI container version (default: from .env or 0.34.0-lmi16.0.0-cu128)"
    )
    create_parser.add_argument(
        "--health-check-timeout",
        type=int,
        default=int(os.getenv("HEALTH_CHECK_TIMEOUT", "1800")),
        help="Health check timeout in seconds (default: from .env or 1800)"
    )
    create_parser.add_argument(
        "--wait",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Wait for endpoint creation to complete (default: false)"
    )
    
    # Delete endpoint command
    delete_parser = subparsers.add_parser('delete', help='Delete an existing endpoint')
    delete_parser.add_argument(
        "--endpoint-name",
        type=str,
        default=os.getenv("ENDPOINT_NAME"),
        help="Endpoint name to delete (default: from .env)"
    )
    delete_parser.add_argument(
        "--region",
        type=str,
        default=os.getenv("AWS_REGION"),
        help="AWS region (default: from .env or default region)"
    )
    
    args = parser.parse_args()
    
    # Validate required arguments for delete command
    if args.command == 'delete' and not args.endpoint_name:
        parser.error("--endpoint-name is required (or set ENDPOINT_NAME in .env)")
    
    if args.command == 'create':
        create_endpoint(
            vllm_config_file=args.vllm_config,
            endpoint_name=args.endpoint_name,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            region=args.region,
            sagemaker_role=args.sagemaker_role,
            container_version=args.container_version,
            health_check_timeout=args.health_check_timeout,
            wait=(args.wait.lower() == "true"),
        )
    elif args.command == 'delete':
        delete_endpoint(
            endpoint_name=args.endpoint_name,
            region=args.region
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
