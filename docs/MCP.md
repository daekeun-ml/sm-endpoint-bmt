# SageMaker Endpoint MCP Server

This MCP server provides SageMaker endpoint metrics checking and benchmarking functionality through MCP (Model Context Protocol).

## Features

### 🔍 Metrics Checking Tools
- `check_endpoint_metrics`: CloudWatch metrics collection and analysis
- `generate_metrics_report`: Generate detailed markdown reports
- `get_endpoint_status`: Query endpoint status information

### 🚀 Benchmarking Tools
- `run_benchmark`: Execute performance benchmarks
- `list_available_datasets`: List available datasets

## Installation and Setup

### 1. Install Dependencies

```bash
# Install FastMCP
pip install fastmcp

# Or use uv (recommended)
uv add fastmcp
```

### 2. MCP Server Configuration

If you cloned this repo, point at `mcp/sm_endpoint_mcp.py` as shown below. If you installed the
package (`pip install 'sm-endpoint-bmt[mcp]'`), call it as a module instead:
`"command": "python", "args": ["-m", "sm_endpoint_mcp"]`.

#### Workspace Level Configuration (.kiro/settings/mcp.json)

```json
{
  "mcpServers": {
    "sm-endpoint-tools": {
      "command": "python",
      "args": ["mcp/sm_endpoint_mcp.py"],
      "cwd": ".",
      "env": {
        "AWS_REGION": "us-east-1"
      },
      "disabled": false,
      "autoApprove": [
        "check_endpoint_metrics",
        "get_endpoint_status",
        "list_available_datasets"
      ]
    }
  }
}
```

#### User Level Configuration (~/.kiro/settings/mcp.json)

```json
{
  "mcpServers": {
    "sm-endpoint-tools": {
      "command": "uvx",
      "args": ["--from", "fastmcp", "python", "/path/to/sm-endpoint-bmt/mcp/sm_endpoint_mcp.py"],
      "env": {
        "AWS_REGION": "us-east-1",
        "AWS_PROFILE": "default"
      },
      "disabled": false,
      "autoApprove": [
        "check_endpoint_metrics",
        "get_endpoint_status"
      ]
    }
  }
}
```

### 3. Environment Variables Setup

Configure environment variables so the MCP server can access AWS:

```bash
# AWS credentials (choose one)
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
# or
export AWS_PROFILE="your-profile"

# AWS region
export AWS_REGION="us-east-1"
```

## Usage

### Metrics Checking

```
Check the metrics for endpoint "my-endpoint" for the last 30 minutes
```

```
Generate a Korean metrics report for "gpt-model-endpoint"
```

```
Show CloudWatch endpoint call history for the last 2 minutes
```

**Note:** CloudWatch metrics are not real-time and may have a 5-15 minute delay.

### Benchmark Execution

```
Run a benchmark with 100 requests on endpoint "my-endpoint"
```

```
Run a performance test with 5 concurrent requests on "llm-endpoint"
```

```
Run a 50-request benchmark using the HuggingFace dataset tatsu-lab/alpaca
```

**Actual Test Results Example (120B model):**
- 50 requests, 2 concurrent users
- Average response time: ~1.3 seconds
- Throughput: 0.75 RPS
- Success rate: 100%

### Endpoint Testing

```
Run a simple call test on SageMaker endpoint gpt-oss-120b-2025-10-16-10-23-39-438
```

### Endpoint Status Check

```
Check the current status of "my-endpoint"
```

## Tool Details

### test_endpoint

Sends a simple test request to a SageMaker endpoint to verify normal operation.

**Parameters:**
- `endpoint_name` (required): SageMaker endpoint name
- `test_payload` (optional): JSON payload for testing
- `region` (optional): AWS region

**Returns:**
- Response time
- Generated text
- Success/failure status

### check_endpoint_metrics

Collects and analyzes SageMaker endpoint metrics from CloudWatch.

**Parameters:**
- `endpoint_name` (required): SageMaker endpoint name
- `minutes` (optional, default: 30): Time range to check (minutes)
- `region` (optional): AWS region
- `language` (optional, default: "en"): Language ("en" or "ko")

**Returns:**
- Metrics summary information
- Detailed metrics data
- Endpoint status information

### generate_metrics_report

Generates a detailed metrics report in markdown format.

**Parameters:**
- `endpoint_name` (required): SageMaker endpoint name
- `minutes` (optional, default: 30): Time range to check (minutes)
- `region` (optional): AWS region
- `language` (optional, default: "en"): Language ("en" or "ko")

**Returns:**
- Detailed report in markdown format
- Metadata

### run_benchmark

Executes performance benchmarks on SageMaker endpoints.

**Parameters:**
- `endpoint_name` (required): SageMaker endpoint name
- `num_requests` (optional, default: 10): Number of requests to send
- `concurrent_requests` (optional, default: 1): Number of concurrent requests
- `region` (optional): AWS region
- `dataset_name` (optional): HuggingFace dataset name (e.g., 'tatsu-lab/alpaca')

**Returns:**
- Benchmark summary results
- Detailed performance metrics
- Latency statistics

### get_endpoint_status

Queries the current status of a SageMaker endpoint.

**Parameters:**
- `endpoint_name` (required): SageMaker endpoint name
- `region` (optional): AWS region

**Returns:**
- Endpoint status information
- Instance information
- Configuration information

### list_available_datasets

Returns a list of datasets available for benchmarking.

**Returns:**
- Built-in dataset list
- HuggingFace dataset usage
- Description of each dataset

## Troubleshooting

### 1. MCP Server Connection Failure

**Symptoms:** Cannot connect to MCP server

**Solutions:**
1. Check path: Verify that `mcp/sm_endpoint_mcp.py` file exists
2. Check permissions: Verify file execution permissions
3. Check dependencies: Ensure FastMCP and required packages are installed

```bash
# Set file permissions
chmod +x mcp/sm_endpoint_mcp.py

# Check dependencies
python -c "import fastmcp; print('FastMCP OK')"
```

### 2. AWS Credentials Error

**Symptoms:** AWS access permission error

**Solutions:**
1. Check AWS credentials:
   ```bash
   aws sts get-caller-identity
   ```

2. Check environment variables:
   ```bash
   echo $AWS_REGION
   echo $AWS_PROFILE
   ```

3. Check IAM permissions:
   - SageMaker read permissions
   - CloudWatch metrics read permissions

### 3. Module Import Error

**Symptoms:** Cannot find `check_metrics` or `sagemaker_benchmark` modules

**Solutions:**
1. Check working directory: Ensure MCP server runs from project root
2. Check path settings: Verify `cwd` configuration is correct
3. Check file existence: Verify required Python files exist

### 4. Performance Optimization

**Slow response times:**
1. Reduce metrics query time range (`minutes` parameter)
2. Reduce benchmark request count (`num_prompts` parameter)
3. Adjust concurrent request count (`concurrent_requests` parameter)

## Security Considerations

1. **Credential Management**: Manage AWS credentials securely
2. **Principle of Least Privilege**: Grant only necessary minimum permissions
3. **Auto Approval**: Add only trusted tools to `autoApprove`

## Development and Extension

### Adding New Tools

```python
@mcp.tool()
def your_new_tool(param1: str, param2: int = 10) -> Dict[str, Any]:
    """
    Your tool description
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
    
    Returns:
        Dictionary containing results
    """
    try:
        # Your implementation here
        result = do_something(param1, param2)
        
        return {
            'success': True,
            'result': result,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
```

### Adding Logging

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use logging in tools
logger.info(f"Running benchmark for endpoint: {endpoint_name}")
```

## References

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [CloudWatch Metrics](https://docs.aws.amazon.com/sagemaker/latest/dg/monitoring-cloudwatch.html)