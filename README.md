# SageMaker Endpoint Benchmark Tool

A SageMaker endpoint benchmarking tool similar to vLLM bench serve.

## Installation

### Using uv (Recommended)

```bash
# Install uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies and create virtual environment
uv sync

# Run scripts (uv automatically activates virtual environment)
uv run python sagemaker_benchmark.py --help
```

### Using pip (Alternative)

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Recommendation**: uv is 10-100x faster and more reliable than pip. It automatically manages virtual environments and has superior dependency resolution. See [uv official documentation](https://docs.astral.sh/uv/) for more details.

## Configuration

### 1. vLLM Configuration File

Copy example files from the `config/` folder to create configuration files:

```bash
# Default model configuration
cp config/vllm_config.json.example config/vllm_config.json

# Or 120B model configuration
cp config/vllm_config_120b.json.example config/vllm_config_120b.json
```

Example `config/vllm_config.json` file:
```json
{
  "HF_MODEL_ID": "openai/gpt-oss-20b",
  "HF_TOKEN": "",
  "OPTION_MODEL_LOADING_TIMEOUT": "1500",
  "SERVING_FAIL_FAST": "true",
  "OPTION_ASYNC_MODE": "true",
  "OPTION_ROLLING_BATCH": "disable",
  "TENSOR_PARALLEL_DEGREE": "max",
  "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service"
}
```

### 2. Environment Variables (Optional)

Copy `.env.example` to create `.env` file and configure instance settings:

```bash
cp .env.example .env
```

Example `.env` file:
```bash
INSTANCE_TYPE=ml.g5.xlarge
INSTANCE_COUNT=1
AWS_REGION=us-east-1
SAGEMAKER_ROLE=arn:aws:iam::YOUR_ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole-XXXXX
VLLM_CONFIG_FILE=config/vllm_config.json
```

**Finding SageMaker Role:**
```bash
# Find SageMaker role using AWS CLI
aws iam list-roles | grep -i sagemaker

# Or AWS Console: IAM > Roles > Search "SageMaker"
```

## Quick Start

### 1. Create Endpoint (Optional)

If you don't have a SageMaker endpoint, create one first:

```bash
# Create with config/vllm_config.json and .env file settings
uv run python create_endpoint.py create

# Use different vLLM configuration file
uv run python create_endpoint.py create --vllm-config config/vllm_config_120b.json

# Override instance type
uv run python create_endpoint.py create --instance-type "ml.g6.48xlarge"
```

### 2. Test Endpoint

Test if the endpoint is working properly:

```bash
# Use ENDPOINT_NAME from .env file
python test_endpoint.py

# Or specify directly
python test_endpoint.py --endpoint-name "your-endpoint-name"
```

### 3. Run Benchmark

```bash
uv run python sagemaker_benchmark.py \
  --endpoint-name 'your-endpoint-name' \
  --num-prompts 200
```

### 4. Advanced Usage

For more examples and parameter descriptions, refer to these documents:
- **[Usage Guide (English)](docs/USAGE.md)** - Detailed usage and examples

## Key Features

- ✅ **vLLM Compatible**: Same CLI interface as vLLM bench serve
- ✅ **Auto Scaling**: CloudWatch metrics-based automatic scaling
- ✅ **MCP Server**: Model Context Protocol server integrated with Kiro IDE
- ✅ **Various Datasets**: Support for Random, ShareGPT, HuggingFace datasets
- ✅ **Accurate Metrics**: Detailed performance metrics including TTFT, TPOT, ITL
- ✅ **Sampling Parameters**: Temperature, top-p, top-k, beam search, etc.
- ✅ **Concurrency Control**: Maximum concurrent requests and request rate control

## Output Metrics

- **Request throughput**: Requests processed per second
- **Token throughput**: Tokens processed per second
- **TTFT (Time to First Token)**: Time until first token
- **TPOT (Time per Output Token)**: Time per output token
- **ITL (Inter-token Latency)**: Latency between tokens

Each metric provides average, median, and P99 values.

## File Structure

```
.
├── config/                              # Configuration files directory
│   ├── vllm_config.json.example         # vLLM configuration example (default)
│   ├── vllm_config_120b.json.example    # vLLM configuration example (120B)
│   └── vllm_config.json                 # Actual vLLM configuration
├── autoscaling/                         # Auto Scaling related tools
│   ├── autoscaling.py                   # Auto Scaling setup script
│   └── test_autoscaling.py              # Auto Scaling test script
├── mcp/                                 # MCP server (Model Context Protocol)
│   ├── sm_endpoint_mcp.py               # SageMaker Endpoint MCP server
│   └── example_mcp_config.json          # MCP configuration example
├── docs/                                # Documentation directory
├── create_endpoint.py                   # Endpoint creation/deletion script
├── sagemaker_benchmark.py               # Main benchmark script
├── benchmark_datasets.py                # Dataset module
├── test_endpoint.py                     # Endpoint test script
├── check_metrics.py                     # CloudWatch metrics checker
├── create_endpoint_tutorial.ipynb       # Endpoint creation tutorial
├── pyproject.toml                       # Project configuration (uv)
├── requirements.txt                     # Required packages (pip)
├── .env.example                         # Environment variables example
└── README.md                            # Basic guide
```

## Endpoint Management

### Creating Endpoints

```bash
# Create with default settings (using config/vllm_config.json)
uv run python create_endpoint.py create

# Specify SageMaker role (required for local environment)
uv run python create_endpoint.py create \
  --sagemaker-role "arn:aws:iam::YOUR_ACCOUNT:role/service-role/AmazonSageMaker-ExecutionRole-XXXXX"

# Use different vLLM configuration file
uv run python create_endpoint.py create --vllm-config config/vllm_config_120b.json

# Override instance settings
uv run python create_endpoint.py create \
  --instance-type "ml.g6.48xlarge" \
  --instance-count 2

# Basic creation (background, no waiting)
uv run python create_endpoint.py create

# Wait until completion
uv run python create_endpoint.py create --wait true
```

### Managing Multiple Model Configurations

You can manage configuration files for various models in the `config/` folder:

```bash
# 20B model
cp config/vllm_config.json.example config/vllm_config_20b.json
nano config/vllm_config_20b.json

# 120B model
cp config/vllm_config_120b.json.example config/vllm_config_120b.json
nano config/vllm_config_120b.json

# Usage
uv run python create_endpoint.py create --vllm-config config/vllm_config_20b.json
uv run python create_endpoint.py create --vllm-config config/vllm_config_120b.json
```

### Deleting Endpoints

```bash
# Use ENDPOINT_NAME from .env file
uv run python create_endpoint.py delete

# Or specify directly
uv run python create_endpoint.py delete --endpoint-name "your-endpoint-name"
```

### Detailed usage

- 📖 **[SageMaker Endpoint Benchmark Tool - Usage Guide](docs/BMT_GUIDE.md)**

### Configuration Priority

1. CLI arguments (highest priority)
2. `.env` file
3. `config/vllm_config.json` file
4. Default values

### Config Folder Structure

You can manage multiple model configurations in the `config/` folder:

```
config/
├── vllm_config.json              # Default configuration
├── vllm_config_20b.json          # 20B model configuration
├── vllm_config_120b.json         # 120B model configuration
└── vllm_config_custom.json       # Custom configuration
```

## Auto Scaling

- 📖 **[Auto Scaling Detailed Guide](docs/AUTOSCALING_GUIDE.md)** - Comprehensive usage guide for auto scaling setup, testing, metrics monitoring, and all features

## MCP Server (Model Context Protocol)

- 📖 **[MCP Server Guide](docs/MCP.md)** - Model Context Protocol server integration guide

## LMI Architecture and vLLM Details

- 📖 **[LMI Architecture Guide](docs/LMI_ARCHITECTURE.md)** - LMI operation principles and architecture guide
- 📖 **[vLLM Configuration Guide](docs/VLLM_CONFIG.md)** - Additional vLLM configuration methods

## AWS Credentials

### 1. AWS Credentials
This tool uses boto3, so AWS credentials are required:

- AWS CLI configuration (`aws configure`)
- Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- IAM role (when running on EC2/ECS)

### 2. SageMaker Execution Role
A SageMaker execution role is required when creating endpoints:

**Running in local environment:**
- Set `SAGEMAKER_ROLE` in `.env` file
- Or use `--sagemaker-role` CLI argument

**Running in SageMaker environment:**
- Automatically detects role in Notebook or Studio

**Role Permissions:**
The SageMaker execution role requires the following permissions:
- SageMaker resource creation/management
- ECR image pull
- S3 access (for model artifacts)

## License

This project is licensed under the MIT License. See the LICENSE file for details.