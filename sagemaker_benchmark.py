"""
SageMaker Endpoint Benchmark Tool
vLLM bench serve와 유사한 벤치마크 도구
"""
import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional
import boto3
import numpy as np
from tqdm.asyncio import tqdm

from benchmark_datasets import get_dataset, SampleRequest

@dataclass
class RequestMetrics:
    """개별 요청의 메트릭 - vLLM RequestFuncOutput과 동일"""
    generated_text: str = ""
    success: bool = False
    total_time: float = 0.0  # latency
    output_len: int = 0  # output_tokens
    ttft: float = 0.0  # Time to first token (s)
    itl: List[float] = field(default_factory=list)  # list of inter-token latencies (s)
    tpot: float = 0.0  # avg next-token latencies (s)
    prompt_len: int = 0
    error: str = ""

@dataclass
class BenchmarkResults:
    """벤치마크 전체 결과"""
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    benchmark_duration: float = 0.0
    request_metrics: List[RequestMetrics] = field(default_factory=list)
    peak_concurrent_requests: int = 0
    
    def calculate_statistics(self):
        """통계 계산 - vLLM과 동일한 방식"""
        if not self.request_metrics:
            return
        
        # TTFT는 초 단위로 저장되어 있으므로 ms로 변환
        ttfts = [m.ttft * 1000 for m in self.request_metrics if m.success and m.ttft > 0]
        
        # ITL도 초 단위로 저장되어 있으므로 ms로 변환
        itls = []
        for m in self.request_metrics:
            if m.success:
                itls.extend([itl * 1000 for itl in m.itl])
        
        # TPOT 계산: vLLM 방식 - (latency - ttft) / (output_len - 1)
        tpots = []
        for m in self.request_metrics:
            if m.success and m.output_len > 1:
                latency_minus_ttft = m.total_time - m.ttft
                tpot = latency_minus_ttft / (m.output_len - 1)
                tpots.append(tpot * 1000)  # ms로 변환
        
        return {
            'ttft': {
                'mean': np.mean(ttfts) if ttfts else 0,
                'median': np.median(ttfts) if ttfts else 0,
                'p99': np.percentile(ttfts, 99) if ttfts else 0,
            },
            'tpot': {
                'mean': np.mean(tpots) if tpots else 0,
                'median': np.median(tpots) if tpots else 0,
                'p99': np.percentile(tpots, 99) if tpots else 0,
            },
            'itl': {
                'mean': np.mean(itls) if itls else 0,
                'median': np.median(itls) if itls else 0,
                'p99': np.percentile(itls, 99) if itls else 0,
            }
        }

class SageMakerBenchmark:
    def __init__(
        self,
        endpoint_name: str,
        region_name: str = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        best_of: int = 1,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
    ):
        self.endpoint_name = endpoint_name
        self.client = boto3.client('sagemaker-runtime', region_name=region_name)
        self.results = BenchmarkResults()
        self.active_requests = 0
        self.max_concurrent = 0
        
        # Sampling parameters
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_beam_search = use_beam_search
        self.best_of = best_of
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

    
    def _sync_request(self, prompt: str, max_tokens: int) -> RequestMetrics:
        """
        동기 방식으로 요청 전송 및 메트릭 수집
        전체 스트림 처리를 동기적으로 수행하여 정확한 타이밍 측정
        """
        output = RequestMetrics(
            prompt_len=len(prompt.split()),
            output_len=0,
            ttft=0.0,
            tpot=0.0
        )
        
        # 샘플링 파라미터 구성
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": True
        }
        
        # top_k는 -1이 아닐 때만 추가
        if self.top_k != -1:
            payload["top_k"] = self.top_k
        
        # best_of는 1보다 클 때만 추가
        if self.best_of > 1:
            payload["best_of"] = self.best_of
        
        # repetition_penalty는 1.0이 아닐 때만 추가
        if self.repetition_penalty != 1.0:
            payload["repetition_penalty"] = self.repetition_penalty
        
        # presence_penalty는 0.0이 아닐 때만 추가
        if self.presence_penalty != 0.0:
            payload["presence_penalty"] = self.presence_penalty
        
        # frequency_penalty는 0.0이 아닐 때만 추가
        if self.frequency_penalty != 0.0:
            payload["frequency_penalty"] = self.frequency_penalty
        
        # beam search 파라미터
        if self.use_beam_search:
            payload["use_beam_search"] = True
            if "best_of" not in payload:
                payload["best_of"] = 1
        
        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        first_chunk_received = False
        buffer = ""  # 불완전한 JSON을 위한 버퍼
        
        try:
            response = self.client.invoke_endpoint_with_response_stream(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            event_stream = response['Body']
            
            for event in event_stream:
                if 'PayloadPart' in event:
                    chunk_bytes = event['PayloadPart']['Bytes']
                    
                    if not chunk_bytes.strip():
                        continue
                    
                    # 버퍼에 추가
                    chunk_str = chunk_bytes.decode('utf-8')
                    buffer += chunk_str
                    
                    # 줄바꿈으로 구분된 JSON 처리
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if not line:
                            continue
                        
                        if line.startswith('data:'):
                            line = line[5:].strip()
                        
                        if line == '[DONE]':
                            continue
                        
                        try:
                            data = json.loads(line)
                            
                            if choices := data.get('choices'):
                                text = choices[0].get('text')
                                timestamp = time.perf_counter()
                                
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    output.ttft = timestamp - st
                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)
                                
                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            
                            if usage := data.get('usage'):
                                output.output_len = usage.get('completion_tokens')
                        
                        except json.JSONDecodeError:
                            # 불완전한 JSON은 무시 (다음 chunk에서 완성될 수 있음)
                            pass
            
            # 남은 버퍼 처리
            if buffer.strip():
                line = buffer.strip()
                if line.startswith('data:'):
                    line = line[5:].strip()
                
                if line and line != '[DONE]':
                    try:
                        data = json.loads(line)
                        if usage := data.get('usage'):
                            output.output_len = usage.get('completion_tokens')
                    except json.JSONDecodeError as e:
                        if not output.error:
                            output.error = f"JSON decode error in final buffer: {str(e)}"
            
            if first_chunk_received:
                output.success = True
            else:
                output.success = False
                if not output.error:
                    output.error = "Never received a valid chunk to calculate TTFT."
            
            # output_len이 설정되지 않았으면 ITL 개수로 계산
            if output.output_len == 0 and first_chunk_received:
                output.output_len = len(output.itl) + 1
            
            output.generated_text = generated_text
            output.total_time = most_recent_timestamp - st
            
        except Exception as e:
            output.success = False
            output.error = f"{type(e).__name__}: {str(e)}"
        
        return output
    
    async def send_request(self, prompt: str, max_tokens: int, request_id: int) -> RequestMetrics:
        """비동기 래퍼: 동기 요청을 executor에서 실행"""
        self.active_requests += 1
        self.max_concurrent = max(self.max_concurrent, self.active_requests)
        
        try:
            loop = asyncio.get_event_loop()
            output = await loop.run_in_executor(
                None,
                lambda: self._sync_request(prompt, max_tokens)
            )
            return output
        finally:
            self.active_requests -= 1
    

    
    async def run_benchmark(
        self,
        requests: List[SampleRequest],
        max_concurrency: int,
        request_rate: float,
    ):
        """벤치마크 실행"""
        num_prompts = len(requests)
        print(f"Starting benchmark with {num_prompts} requests...")
        print(f"Max concurrency: {max_concurrency}")
        print(f"Request rate: {'inf' if request_rate == float('inf') else f'{request_rate} req/s'}")
        print()
        
        start_time = time.time()
        
        # 세마포어로 동시성 제어
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def limited_request(request: SampleRequest):
            async with semaphore:
                return await self.send_request(
                    request.prompt,
                    request.expected_output_len,
                    int(request.request_id) if request.request_id else 0
                )
        
        # 요청 전송
        if request_rate == float('inf'):
            # 무제한 속도로 전송
            tasks = [limited_request(req) for req in requests]
            metrics_list = await tqdm.gather(*tasks, desc="Sending requests")
        else:
            # 지정된 속도로 전송
            interval = 1.0 / request_rate
            tasks = []
            for i, req in enumerate(requests):
                tasks.append(limited_request(req))
                if i < num_prompts - 1:
                    await asyncio.sleep(interval)
            
            metrics_list = await tqdm.gather(*tasks, desc="Sending requests")
        
        end_time = time.time()
        
        # 결과 집계
        self.results.benchmark_duration = end_time - start_time
        self.results.request_metrics = metrics_list
        self.results.peak_concurrent_requests = self.max_concurrent
        
        for metrics in metrics_list:
            if metrics.success:
                self.results.successful_requests += 1
                self.results.total_input_tokens += metrics.prompt_len
                self.results.total_output_tokens += metrics.output_len
            else:
                self.results.failed_requests += 1
    
    def print_results(self):
        """결과 출력"""
        stats = self.results.calculate_statistics()
        
        print("\n" + "="*50)
        print("Serving Benchmark Result".center(50))
        print("="*50)
        print(f"Successful requests:                {self.results.successful_requests}")
        print(f"Failed requests:                    {self.results.failed_requests}")
        
        # 실패한 요청의 에러 출력
        if self.results.failed_requests > 0:
            print("\n" + "-"*50)
            print("Failed Request Errors (first 3):")
            print("-"*50)
            failed_count = 0
            for i, m in enumerate(self.results.request_metrics):
                if not m.success and failed_count < 3:
                    print(f"\nRequest {i}: {m.error[:200]}")
                    failed_count += 1
        print(f"Maximum request concurrency:        {self.max_concurrent}")
        print(f"Benchmark duration (s):             {self.results.benchmark_duration:.2f}")
        print(f"Total input tokens:                 {self.results.total_input_tokens}")
        print(f"Total generated tokens:             {self.results.total_output_tokens}")
        
        if self.results.benchmark_duration > 0:
            req_throughput = self.results.successful_requests / self.results.benchmark_duration
            output_throughput = self.results.total_output_tokens / self.results.benchmark_duration
            total_throughput = (self.results.total_input_tokens + self.results.total_output_tokens) / self.results.benchmark_duration
            
            print(f"Request throughput (req/s):         {req_throughput:.2f}")
            print(f"Output token throughput (tok/s):    {output_throughput:.2f}")
            print(f"Total Token throughput (tok/s):     {total_throughput:.2f}")
        
        if stats:
            print("\n" + "-"*50)
            print("Time to First Token".center(50))
            print("-"*50)
            print(f"Mean TTFT (ms):                     {stats['ttft']['mean']:.2f}")
            print(f"Median TTFT (ms):                   {stats['ttft']['median']:.2f}")
            print(f"P99 TTFT (ms):                      {stats['ttft']['p99']:.2f}")
            
            print("\n" + "-"*50)
            print("Time per Output Token (excl. 1st token)".center(50))
            print("-"*50)
            print(f"Mean TPOT (ms):                     {stats['tpot']['mean']:.2f}")
            print(f"Median TPOT (ms):                   {stats['tpot']['median']:.2f}")
            print(f"P99 TPOT (ms):                      {stats['tpot']['p99']:.2f}")
            
            print("\n" + "-"*50)
            print("Inter-token Latency".center(50))
            print("-"*50)
            print(f"Mean ITL (ms):                      {stats['itl']['mean']:.2f}")
            print(f"Median ITL (ms):                    {stats['itl']['median']:.2f}")
            print(f"P99 ITL (ms):                       {stats['itl']['p99']:.2f}")
        
        print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="SageMaker Endpoint Benchmark Tool (vLLM-style)"
    )
    
    # Endpoint arguments
    parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="SageMaker endpoint name"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="AWS region (default: use default region)"
    )
    
    # Dataset arguments
    dataset_group = parser.add_argument_group("dataset arguments")
    dataset_group.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["random", "sharegpt", "huggingface", "hf"],
        help="Dataset to use for benchmarking (default: random)"
    )
    dataset_group.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset file (sharegpt) or HuggingFace dataset ID (huggingface)"
    )
    dataset_group.add_argument(
        "--num-prompts",
        type=int,
        default=200,
        help="Number of prompts to process (default: 200)"
    )
    dataset_group.add_argument(
        "--disable-shuffle",
        action="store_true",
        help="Disable shuffling of dataset samples"
    )
    dataset_group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset sampling (default: 0)"
    )
    
    # HuggingFace dataset arguments
    hf_group = parser.add_argument_group("HuggingFace dataset arguments")
    hf_group.add_argument(
        "--hf-prompt-column",
        type=str,
        default="prompt",
        help="Column name for prompts in HuggingFace dataset (default: prompt)"
    )
    hf_group.add_argument(
        "--hf-completion-column",
        type=str,
        default="completion",
        help="Column name for completions in HuggingFace dataset (default: completion)"
    )
    
    # Random dataset arguments
    random_group = parser.add_argument_group("random dataset arguments")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Random input length in tokens (default: 1024)"
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Random output length in tokens (default: 128)"
    )
    
    # Benchmark arguments
    benchmark_group = parser.add_argument_group("benchmark arguments")
    benchmark_group.add_argument(
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent requests (default: 10)"
    )
    benchmark_group.add_argument(
        "--request-rate",
        type=str,
        default="inf",
        help="Request rate in requests/second. Use 'inf' for unlimited (default: inf)"
    )
    
    # Sampling parameters
    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter (default: 1.0)"
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling parameter. -1 means disabled (default: -1)"
    )
    sampling_group.add_argument(
        "--use-beam-search",
        action="store_true",
        help="Use beam search instead of sampling"
    )
    sampling_group.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Number of sequences to generate and return the best one (default: 1)"
    )
    sampling_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (default: 1.0)"
    )
    sampling_group.add_argument(
        "--presence-penalty",
        type=float,
        default=0.0,
        help="Presence penalty (default: 0.0)"
    )
    sampling_group.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.0,
        help="Frequency penalty (default: 0.0)"
    )
    
    args = parser.parse_args()
    
    # request_rate 파싱
    request_rate = float('inf') if args.request_rate == 'inf' else float(args.request_rate)
    
    # 데이터셋 로드
    print(f"Loading dataset: {args.dataset_name}")
    dataset = get_dataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        seed=args.seed,
        disable_shuffle=args.disable_shuffle,
        hf_prompt_column=args.hf_prompt_column,
        hf_completion_column=args.hf_completion_column,
    )
    
    # 샘플 생성
    requests = dataset.sample(
        num_requests=args.num_prompts,
        input_len=args.random_input_len,
        output_len=args.random_output_len,
    )
    print(f"Generated {len(requests)} requests")
    
    # 벤치마크 실행
    benchmark = SageMakerBenchmark(
        endpoint_name=args.endpoint_name,
        region_name=args.region,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_beam_search=args.use_beam_search,
        best_of=args.best_of,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
    )
    
    asyncio.run(
        benchmark.run_benchmark(
            requests=requests,
            max_concurrency=args.max_concurrency,
            request_rate=request_rate,
        )
    )
    
    benchmark.print_results()

if __name__ == "__main__":
    main()
