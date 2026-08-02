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
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Default LMI container. 0.36.0-lmi26.0.0-cu130 is what
# image_uris.retrieve(framework="djl-lmi", version="latest") resolves to; it ships a
# far newer vLLM than the old 0.34.0-lmi16.0.0-cu128 default (vLLM 0.10.2).
# NOTE: LMI v17+ makes async mode the default, so OPTION_ASYNC_MODE /
# OPTION_ROLLING_BATCH=disable / OPTION_ENTRYPOINT in the config examples become
# legacy no-ops on this tag.
DEFAULT_CONTAINER_VERSION = "0.36.0-lmi26.0.0-cu130"

# 독립 vLLM DLC. LMI 와 달리 SM_VLLM_* env 를 읽고, vLLM 을 그대로 최신 버전으로 쓴다.
# gemma-4 는 vLLM >= 0.19 가 필요해서 구 LMI 로는 로드되지 않는다 — 그래서 별도 경로가 필요하다.
# 태그는 https://aws.github.io/deep-learning-containers/reference/available_images/ 에서 재확인.
DEFAULT_VLLM_CONTAINER_VERSION = "0.26.0-gpu-py312-cu130-ubuntu22.04-sagemaker"


def _resolve_sagemaker_sdk() -> dict:
    """
    Resolve the SageMaker SDK symbols this script needs across v2 and v3.

    v3 removed the entire top-level v2 surface (`sagemaker.Model`,
    `sagemaker.session.Session`, `sagemaker.get_execution_role`,
    `sagemaker.utils.name_from_base` all raise ImportError/AttributeError), so the
    call sites must not name a single layout.
    """
    import sagemaker

    resolved: dict = {}

    # --- Session / role / naming -------------------------------------------
    try:  # v2
        from sagemaker.session import Session
        from sagemaker.utils import name_from_base

        resolved["Session"] = Session
        resolved["get_execution_role"] = sagemaker.get_execution_role
        resolved["name_from_base"] = name_from_base
        resolved["version"] = 2
    except (ImportError, AttributeError):  # v3
        from sagemaker.core.helper.session_helper import Session, get_execution_role
        from sagemaker.core.utils import name_from_base

        resolved["Session"] = Session
        resolved["get_execution_role"] = get_execution_role
        resolved["name_from_base"] = name_from_base
        resolved["version"] = 3

    # --- image_uris ---------------------------------------------------------
    try:
        from sagemaker import image_uris  # v2
    except ImportError:
        try:
            from sagemaker.core import image_uris  # v3
        except ImportError:
            image_uris = None
    resolved["image_uris"] = image_uris

    # --- Model builder + deploy kwarg name ----------------------------------
    if resolved["version"] == 2:
        from sagemaker.model import Model

        def build_model(image_uri, role, env, session, instance_type):
            return Model(
                image_uri=image_uri, role=role, env=env, sagemaker_session=session
            )

        resolved["container_timeout_kwarg"] = (
            "container_startup_health_check_timeout"
        )
    else:
        from sagemaker.serve import ModelBuilder

        def build_model(image_uri, role, env, session, instance_type):
            # v3 renamed `env` to `env_vars` and split model creation from deploy.
            #
            # 🔴 build() 의 반환값에 deploy() 를 호출하면 안 된다. 단일 모델 경로에서
            #    build() 는 sagemaker.core.resources.Model 을 돌려주고 그 클래스에는
            #    deploy 가 없다(deploy 는 ModelBuilder 에만 있다). SDK docstring 의
            #    용법도 model = mb.build(); endpoint = mb.deploy() 다. 그래서 build()
            #    는 부수효과로만 쓰고, 배포는 builder 를 통해 한다.
            builder = ModelBuilder(
                image_uri=image_uri,
                env_vars=env,
                role_arn=role,
                instance_type=instance_type,
            )
            builder.build()
            return builder

        resolved["container_timeout_kwarg"] = "container_timeout_in_seconds"

    resolved["build_model"] = build_model
    return resolved


def _detect_engine(vllm_config: dict) -> str:
    """config 의 env 키로 서빙 엔진을 판별한다.

    두 컨테이너는 서로의 env 를 읽지 않는다. LMI 용 OPTION_* 를 vLLM DLC 에 주면 조용히
    무시되고 기본값으로 뜨므로, 사용자가 --engine 을 잊었을 때 이 판별이 사고를 막는다.
    """
    keys = [k for k in vllm_config if not k.startswith("_")]
    if any(k.startswith("SM_VLLM_") for k in keys):
        return "vllm"
    if any(k.startswith("SM_SGLANG_") for k in keys):
        return "sglang"
    return "lmi"


def _resolve_container_uri(sdk: dict, region: str, container_version: str,
                           engine: str = "lmi") -> str:
    """
    Build the serving container URI for `region`.

    engine="lmi"  -> djl-inference (reads OPTION_* env)
    engine="vllm" -> vllm          (reads SM_VLLM_* env)

    The two containers are not interchangeable: their env keys differ, and a config written
    for one will be silently ignored by the other. gemma-4 needs vLLM >= 0.19, which older
    LMI images do not ship, so the vllm path is not optional for those models.

    Prefer the SDK's registry map, which knows the per-region DLC account id and the
    correct DNS suffix. Only fall back to the us-* account when the SDK cannot help,
    and say so, because that fallback is wrong outside the standard partition.
    """
    if engine == "vllm":
        # vLLM DLC 는 SDK 레지스트리에 없다(image_uris.retrieve 가 'vllm' framework 를 모른다).
        # 그래서 패턴으로 조립한다. 계정 763104351884 는 표준 파티션의 DLC 계정이다.
        return (f"763104351884.dkr.ecr.{region}.amazonaws.com/"
                f"vllm:{container_version}")

    image_uris = sdk.get("image_uris")
    if image_uris is not None:
        try:
            # container_version is a full tag ("0.36.0-lmi26.0.0-cu130"); the registry
            # is keyed by the DJL version prefix.
            djl_version = container_version.split("-")[0]
            uri = image_uris.retrieve(
                framework="djl-lmi", region=region, version=djl_version
            )
            # Keep the caller's exact tag rather than the registry's default tag.
            return f"{uri.split(':')[0]}:{container_version}"
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ image_uris.retrieve failed ({e}); falling back to the "
                  "standard-partition DLC account")

    print("⚠️ Using the us-* DLC account id, which is incorrect in regions such as "
          "ap-east-1, cn-*, us-gov-*, il-central-1 and me-central-1. Verify the URI "
          "or upgrade the SageMaker SDK.")
    return (
        f"763104351884.dkr.ecr.{region}.amazonaws.com/"
        f"djl-inference:{container_version}"
    )


def load_vllm_config(config_file: str) -> dict:
    """Load vLLM configuration from JSON file"""
    if not os.path.exists(config_file):
        raise FileNotFoundError(
            f"vLLM config file not found: {config_file}\n"
            f"Please create a config file in the config/ directory.\n"
            f"Example: cp config/lmi/gpt-oss-20b.json.example {config_file}"
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
    container_version: str = DEFAULT_CONTAINER_VERSION,
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
    # Import sagemaker only when needed for create.
    #
    # The SDK moved every symbol used here between v2 and v3, so resolve them at
    # runtime instead of assuming one layout:
    #   v2: sagemaker.Model / sagemaker.session.Session / sagemaker.get_execution_role
    #       / sagemaker.utils.name_from_base / sagemaker.image_uris
    #   v3: sagemaker.serve.ModelBuilder / sagemaker.core.helper.session_helper.*
    #       / sagemaker.core.utils.name_from_base / sagemaker.core.image_uris
    sdk = _resolve_sagemaker_sdk()
    
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
    boto_session = boto3.Session(region_name=region)
    sess = sdk["Session"](boto_session)
    # Do not rely on an SDK-version-specific attribute for the region.
    region = region or boto_session.region_name
    
    # Get SageMaker execution role
    if sagemaker_role:
        role = sagemaker_role
        print(f"Using provided role: {role}")
    else:
        try:
            role = sdk["get_execution_role"]()
            print(f"Auto-detected role: {role}")
        except Exception:
            # Outside SageMaker this raises ValueError, but a missing/moved SDK
            # symbol raises AttributeError - catching only ValueError made the
            # IAM fallback below unreachable.
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
    
    # Resolve the container URI through the SDK's registry map. The DLC account id
    # differs per region (ap-east-1, cn-*, us-gov-*, il-central-1, ... all differ) and
    # China regions need the .amazonaws.com.cn suffix, so hardcoding one account and
    # suffix produces an invalid URI in roughly 20 regions.
    engine = _detect_engine(vllm_config)
    if engine == "vllm" and container_version == DEFAULT_CONTAINER_VERSION:
        # LMI 태그를 vLLM DLC 에 붙이면 존재하지 않는 이미지가 된다. 엔진에 맞는 기본값으로 바꾼다.
        container_version = DEFAULT_VLLM_CONTAINER_VERSION
    print(f"Serving engine: {engine} (detected from config env keys)")
    container_uri = _resolve_container_uri(sdk, region, container_version, engine=engine)
    print(f"Container URI: {container_uri}")
    
    print("\nvLLM Environment Configuration:")
    for key, value in vllm_config.items():
        if key == "HF_TOKEN" and value:
            print(f"  {key}: ***")
        else:
            print(f"  {key}: {value}")
    print()
    
    # Create model. v3 renamed `env` to `env_vars` and replaced Model with
    # ModelBuilder, so build through the resolved factory.
    model = sdk["build_model"](
        image_uri=container_uri,
        role=role,
        env=vllm_config,
        session=sess,
        instance_type=instance_type,
    )
    
    # Generate endpoint name if not provided
    if endpoint_name is None or endpoint_name == "":
        model_name = model_id.split('/')[-1]
        endpoint_name = sdk["name_from_base"](model_name)
    
    print(f"Endpoint Name: {endpoint_name}")
    print()
    
    # Deploy model
    print("Deploying model...")
    print("This may take 10-20 minutes depending on the model size.")
    print()
    
    try:
        # The container cold-start timeout kwarg was renamed in v3
        # (container_startup_health_check_timeout -> container_timeout_in_seconds).
        # Large models need it either way, so pass whichever the SDK accepts.
        deploy_kwargs = {
            "initial_instance_count": instance_count,
            "instance_type": instance_type,
            "endpoint_name": endpoint_name,
            "wait": wait,
            sdk["container_timeout_kwarg"]: health_check_timeout,
        }
        predictor = model.deploy(**deploy_kwargs)
        
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
    
    # botocore models NO error shapes for DeleteEndpoint / DeleteEndpointConfig, so
    # a missing resource arrives as a generic ClientError whose code is not
    # guaranteed to be ValidationException. Treat any "not there" code as success.
    not_found_codes = {'ValidationException', 'ResourceNotFound', 'ResourceNotFoundException'}

    # Delete endpoint
    try:
        client.delete_endpoint(EndpointName=endpoint_name)
        print(f"✅ Endpoint deleted: {endpoint_name}")
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') in not_found_codes:
            print(f"⚠️ Endpoint not found: {endpoint_name}")
        else:
            print(f"⚠️ Error deleting endpoint: {e}")
    
    # Delete endpoint configuration
    try:
        client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"✅ Endpoint config deleted: {endpoint_name}")
    except ClientError as e:
        if e.response.get('Error', {}).get('Code') in not_found_codes:
            print(f"⚠️ Endpoint config not found: {endpoint_name}")
        else:
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
        default=os.getenv("VLLM_CONFIG_FILE", "config/vllm/gemma-4-E4B.json"),
        help="Path to serving config JSON. config/vllm/*(SM_VLLM_*) or config/lmi/*(OPTION_*) — "
             "the engine is detected from the env keys inside "
             "(default: from .env or config/vllm/gemma-4-E4B.json)"
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
        default=os.getenv("CONTAINER_VERSION", DEFAULT_CONTAINER_VERSION),
        help=f"LMI container version (default: from .env or {DEFAULT_CONTAINER_VERSION})"
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
