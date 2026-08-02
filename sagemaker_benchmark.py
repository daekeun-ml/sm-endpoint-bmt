"""
SageMaker Endpoint Benchmark Tool

`vllm bench serve`와 동일한 메트릭/CLI를 SageMaker 실시간 엔드포인트에 적용한다.
동일한 엔드포인트·동일한 부하에서 `vllm bench serve` 사용자가 알아볼 수 있는 숫자가
나오는 것이 이 파일의 목표다.

parity 기준 (2026-08 기준 vLLM main):
  - vllm/benchmarks/serve.py                     : 지표 공식, pacing, 리포트 레이아웃
  - vllm/benchmarks/lib/endpoint_request_func.py : TTFT/ITL/E2EL 측정 지점, SSE 파싱
  - vllm/benchmarks/lib/ready_checker.py         : ready check

SageMaker 고유 부분 (vLLM에 대응물이 없어 그대로 유지/확장한 것):
  - transport 는 HTTP/aiohttp 가 아니라 boto3 `invoke_endpoint_with_response_stream`
    (SigV4 서명 + eventstream 프레이밍). blocking 호출이라 thread executor 로 offload 한다.
  - PayloadPart 경계는 SSE 경계와 무관하므로 바이트 단위 누적 버퍼가 필수다.
  - botocore Config(재시도/타임아웃/커넥션 풀)가 vLLM 의 TCPConnector 설정을 대체한다.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import contextlib
import json
import os
import sys
import time
import traceback
import uuid
import warnings
from collections.abc import AsyncGenerator, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import boto3
import numpy as np
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from tqdm.asyncio import tqdm

from benchmark_datasets import SampleRequest, get_dataset

# vLLM serve.py:59 과 동일한 상수. goodput SLO 는 ms 로 받고 내부는 초 단위다.
MILLISECONDS_TO_SECONDS_CONVERSION = 1000

# boto3 는 async 클라이언트가 없다. 동시성 = 스레드 수이므로, 스레드가 너무 많아지면
# GIL 경합이 ITL 측정을 흔든다. 그 임계값을 넘으면 경고만 하고 진행한다.
THREAD_COUNT_WARN_THRESHOLD = 256

# SageMaker 컨테이너 스트리밍은 최초 토큰까지 오래 걸릴 수 있다(콜드 컨테이너, 큐 대기).
# botocore 기본 read_timeout 60s 는 이런 요청을 끊어버리므로 넉넉하게 잡는다.
DEFAULT_READ_TIMEOUT_S = 900
DEFAULT_CONNECT_TIMEOUT_S = 10


# ---------------------------------------------------------------------------
# vLLM RequestFuncInput / RequestFuncOutput 미러
# ---------------------------------------------------------------------------


@dataclass
class RequestFuncInput:
    """vLLM `RequestFuncInput` 미러. api_url 대신 endpoint_name 을 쓴다."""

    prompt: str
    prompt_len: int
    output_len: int
    endpoint_name: str
    model: str | None = None
    request_id: str | None = None
    ignore_eos: bool = False
    extra_body: dict[str, Any] | None = None
    # "openai"(=completions) 또는 "openai-chat". vLLM 의 --backend 와 같은 의미.
    endpoint_type: str = "openai"
    include_usage: bool = True


@dataclass
class RequestFuncOutput:
    """vLLM `RequestFuncOutput` 미러 + SageMaker 전용 진단 필드."""

    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token (s)
    itl: list[float] = field(default_factory=list)  # inter-token latencies (s)
    tpot: float = 0.0  # vLLM 과 동일하게 여기서는 채우지 않는다(serve.py 에서 계산).
    prompt_len: int = 0
    error: str = ""
    start_time: float = 0.0

    # --- 이하 SageMaker 전용 (vLLM 에는 없음) ---
    # finish_reason 을 기록해야 "요청한 길이만큼 생성됐는지"를 판별할 수 있다.
    finish_reason: str | None = None
    # 서버가 usage 프레임을 주지 않으면 output_tokens 를 신뢰할 수 없다는 표시.
    usage_reported: bool = False
    # 파싱 불가 프레임 수. 0 이 아니면 토큰 손실이 있었다는 뜻이므로 실패로 처리한다.
    dropped_frames: int = 0
    # ThrottlingException / ModelError / ReadTimeoutError 등 분류용.
    error_kind: str = ""


@dataclass
class BenchmarkMetrics:
    """vLLM `BenchmarkMetrics` 와 동일한 필드 집합 (+ SageMaker 확장)."""

    completed: int
    failed: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]
    max_output_tokens_per_s: float
    max_concurrent_requests: int

    # --- SageMaker 확장 -----------------------------------------------------
    # finish_reason 분해: 잘림(length)은 오류가 아니라 "결과"다.
    truncated_requests: int = 0
    eos_stopped_requests: int = 0
    unknown_finish_requests: int = 0
    # usage 프레임을 못 받은 요청 수. 이 값이 0 이 아니면 출력 토큰 수는 추정치다.
    requests_without_usage: int = 0
    # 출력 토큰 수의 출처: "server-usage" / "tokenizer" / "fallback-1"
    output_len_source: str = "server-usage"
    # 오류 종류별 카운트 (ThrottlingException, ModelError, ...)
    error_kinds: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SSE / eventstream 파싱
# ---------------------------------------------------------------------------


class StreamedResponseHandler:
    """
    PayloadPart 바이트 스트림을 완결된 메시지 리스트로 바꿔주는 누적 버퍼.

    왜 필요한가 (SageMaker 고유 문제):
      SageMaker 는 컨테이너가 쓴 바이트를 eventstream 프레임(PayloadPart)으로 재분할한다.
      그 경계는 SSE 줄/프레임 경계와 아무 관련이 없어서, 한 PayloadPart 가 JSON 중간이나
      멀티바이트 UTF-8 시퀀스 중간에서 끊길 수 있다. 따라서
        (1) 디코딩은 incremental codec 으로 (바이트 경계 안전), vLLM StreamedResponseHandler
            와 동일한 방식이고,
        (2) 프레임 조립은 누적 버퍼에서만 수행해야 한다.
      PayloadPart 단위로 decode() 하거나 공백 PayloadPart 를 버퍼에 넣기 전에 건너뛰면
      줄 경계가 파괴되어 토큰이 조용히 사라진다.

    프레이밍은 두 가지를 모두 받아준다 (SageMaker 컨테이너들이 서로 다르다):
      - OpenAI 호환 SSE : "data: {...}\\n\\n" (+ "data: [DONE]")
      - LMI/TGI NDJSON  : "{...}\\n"
    """

    def __init__(self) -> None:
        self.buffer = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        # 줄바꿈이 포함된(=아직 완성되지 않은) 프레임을 이어붙이기 위한 보관함.
        self._pending = ""

    @staticmethod
    def _payload_of(message: str) -> str:
        """SSE 필드 접두사를 제거한 실제 payload."""
        if message.startswith("data:"):
            return message[len("data:") :].strip()
        return message

    @classmethod
    def _is_complete(cls, message: str) -> bool:
        # SSE 주석/keepalive(":" 로 시작)는 그 자체로 완결된 메시지다.
        if message.startswith(":"):
            return True
        payload = cls._payload_of(message)
        if not payload or payload == "[DONE]":
            return True
        try:
            json.loads(payload)
        except json.JSONDecodeError:
            return False
        return True

    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        """PayloadPart 바이트를 넣고, 완결된 메시지들을 순서대로 반환한다."""
        self.buffer += self._decoder.decode(chunk_bytes)

        messages: list[str] = []
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.strip()

            if not line:
                # 빈 줄 = SSE 프레임 종료자. 보류 중인 조각이 있으면 여기서 확정한다.
                if self._pending:
                    messages.append(self._pending)
                    self._pending = ""
                continue

            candidate = self._pending + line if self._pending else line
            if self._is_complete(candidate):
                messages.append(candidate)
                self._pending = ""
            else:
                # JSON 이 아직 안 닫혔다 → 다음 줄과 합친다.
                self._pending = candidate

        return messages

    def flush(self) -> list[str]:
        """스트림 종료 시 남은 조각을 마지막 메시지로 확정한다."""
        self.buffer += self._decoder.decode(b"", final=True)
        tail = (self._pending + self.buffer).strip()
        self._pending = ""
        self.buffer = ""
        return [tail] if tail else []


def _extract_delta(data: dict[str, Any]) -> tuple[bool, str, str | None]:
    """
    한 프레임에서 (콘텐츠 프레임인지, 델타 텍스트, finish_reason) 을 뽑아낸다.

    SageMaker 컨테이너별 스키마가 달라서 vLLM 처럼 choices 하나만 볼 수 없다.
    지원 스키마:
      - OpenAI completions : {"choices":[{"text": "...", "finish_reason": ...}]}
      - OpenAI chat        : {"choices":[{"delta":{"content": "..."}}]}
      - LMI/TGI            : {"token":{"text": "..."}, "details":{"finish_reason":...}}
      - DJL rolling batch  : {"outputs":["..."]}
      - 비스트리밍 마무리   : {"generated_text": "..."}

    vLLM 과 동일하게 "텍스트가 비어 있어도 콘텐츠 프레임이면 TTFT/ITL 경계로 센다"
    (특수 토큰이 빈 문자열로 오는 경우가 있음, endpoint_request_func.py:226-243).
    """
    if choices := data.get("choices"):
        choice = choices[0] or {}
        finish_reason = choice.get("finish_reason") or choice.get("stop_reason")
        if "text" in choice:  # completions
            return True, choice.get("text") or "", finish_reason
        delta = choice.get("delta")
        if isinstance(delta, dict):  # chat completions
            text = delta.get("content")
            if text is None:
                # reasoning 모델은 content 없이 reasoning_content 만 보낼 수 있다.
                text = delta.get("reasoning_content")
            return True, text or "", finish_reason
        if "message" in choice:  # 비스트리밍 응답이 섞여 들어온 경우
            message = choice.get("message") or {}
            return True, message.get("content") or "", finish_reason
        return True, "", finish_reason

    if isinstance(token := data.get("token"), dict):  # TGI / LMI
        details = data.get("details") or {}
        return True, token.get("text") or "", details.get("finish_reason")

    if isinstance(outputs := data.get("outputs"), list):  # DJL rolling batch
        details = data.get("details") or {}
        return True, "".join(str(o) for o in outputs), details.get("finish_reason")

    if "generated_text" in data and "usage" not in data:
        details = data.get("details") or {}
        return True, data.get("generated_text") or "", details.get("finish_reason")

    # 콘텐츠는 없지만 finish_reason 만 실린 마무리 프레임 (LMI/TGI 가 이렇게 보낸다).
    if isinstance(details := data.get("details"), dict):
        return False, "", details.get("finish_reason")
    if isinstance(finish_reason := data.get("finish_reason"), str):
        return False, "", finish_reason

    return False, "", None


def _extract_usage(data: dict[str, Any]) -> tuple[int | None, int | None]:
    """(completion_tokens, prompt_tokens) 을 반환. 없으면 None."""
    usage = data.get("usage")
    if isinstance(usage, dict):
        ct = usage.get("completion_tokens")
        pt = usage.get("prompt_tokens")
        return (
            int(ct) if isinstance(ct, (int, float)) else None,
            int(pt) if isinstance(pt, (int, float)) else None,
        )
    # TGI/LMI 는 details 안에 토큰 수를 넣는다.
    details = data.get("details")
    if isinstance(details, dict):
        ct = details.get("generated_tokens")
        pt = details.get("prompt_tokens")
        return (
            int(ct) if isinstance(ct, (int, float)) else None,
            int(pt) if isinstance(pt, (int, float)) else None,
        )
    return None, None


class SageMakerStreamError(RuntimeError):
    """
    200 응답 이후 스트림 중간에 도착하는 실패 이벤트.

    vLLM 의 모델에는 이런 상태가 없다(HTTP 헤더가 200 이면 성공). SageMaker 는
    ModelStreamError(400, ModelInvocationTimeExceeded / StreamBroken) 와
    InternalStreamFailure(500) 를 eventstream 안에서 보내므로 별도 타입으로 분류한다.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def _classify_error(exc: BaseException) -> str:
    """
    SageMaker 오류 분류. vLLM 에는 없는 개념이지만, 429 스로틀과 모델 크래시를
    구분하지 못하면 나쁜 런의 원인을 설명할 수 없다.
    """
    if isinstance(exc, SageMakerStreamError):
        return exc.kind
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code") or ""
        return code or "ClientError"
    return type(exc).__name__


# ---------------------------------------------------------------------------
# SageMaker transport
# ---------------------------------------------------------------------------


def build_boto_config(
    max_concurrency: int,
    read_timeout: float,
    connect_timeout: float,
) -> BotoConfig:
    """
    botocore Config 는 vLLM 의 aiohttp TCPConnector 설정 자리다. 명시적으로 고정하는 이유:

    - retries total_max_attempts=1 : 기본 legacy 모드는 503/스로틀/ReadTimeout 을 최대 5회
      조용히 재전송한다. 그러면 (a) 엔드포인트에 의도한 것보다 많은 부하가 가고,
      (b) 재시도 시간이 한 요청의 TTFT/E2EL 에 합산되어 P99 를 오염시킨다.
      재시도는 하네스가 관측·보고해야 할 대상이므로 클라이언트 자동 재시도는 끈다.
    - max_pool_connections >= 동시성 : 풀이 작으면 초과 커넥션마다 TCP+TLS 핸드셰이크를
      새로 하고 버려서 TTFT 에 세팅 비용이 섞인다(요청이 큐잉되지는 않는다).
    - read_timeout 은 소켓 read 당 타임아웃이다. 긴 생성 자체는 문제되지 않지만
      콜드 컨테이너의 first-byte 지연이 기본 60s 를 넘으면 요청이 끊긴다.
    """
    return BotoConfig(
        retries={"mode": "standard", "total_max_attempts": 1},
        max_pool_connections=max(max_concurrency, 10),
        read_timeout=read_timeout,
        connect_timeout=connect_timeout,
        tcp_keepalive=True,
    )


def build_payload(request_func_input: RequestFuncInput) -> dict[str, Any]:
    """
    /invocations 페이로드 조립.

    vLLM 이 항상 보내는 두 가지를 반드시 포함한다:
      - "stream": True
      - "stream_options": {"include_usage": True}  → 서버가 보고한 completion_tokens/
        prompt_tokens 를 받기 위한 것. 이게 없으면 출력 토큰 수를 청크 수로 추정하게 되고
        TPOT 가 mean(ITL) 로 붕괴한다(serve.py:589-605 의 경고와 동일한 함정).
    나머지 샘플링 파라미터는 vLLM 처럼 명시된 값만 extra_body 로 실어 보낸다. 컨테이너
    (LMI/DJL/TGI/vLLM-on-SM)마다 허용 키가 달라서 기본값 주입은 하지 않는다.
    """
    if request_func_input.endpoint_type == "openai-chat":
        payload: dict[str, Any] = {
            "messages": [{"role": "user", "content": request_func_input.prompt}],
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
    else:
        payload = {
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }

    if request_func_input.include_usage:
        payload["stream_options"] = {"include_usage": True}
    if request_func_input.model:
        payload["model"] = request_func_input.model
    if request_func_input.ignore_eos:
        # random 데이터셋에서 요청한 길이만큼 실제로 생성되게 한다(vLLM serve.py:2091-2110).
        payload["ignore_eos"] = True
    if request_func_input.extra_body:
        payload.update(request_func_input.extra_body)
    return payload


def _blocking_stream_request(
    client: Any,
    request_func_input: RequestFuncInput,
    st: float,
) -> RequestFuncOutput:
    """
    boto3 스트리밍 호출 (blocking). 반드시 executor 스레드에서 실행되어야 한다.

    `st` 는 호출 코루틴에서 측정해 전달받는다. 그래야 executor 큐 대기 시간이
    지연에 포함되어(=숨지 않고) 동시성이 실제로 확보됐는지 드러난다.
    """
    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len
    output.start_time = st

    payload = build_payload(request_func_input)
    generated_text = ""
    most_recent_timestamp = st
    first_chunk_received = False
    handler = StreamedResponseHandler()

    def consume(messages: list[str]) -> None:
        nonlocal generated_text, most_recent_timestamp, first_chunk_received
        for message in messages:
            # SSE 주석(핑)은 데이터가 아니다 (endpoint_request_func.py:214-218).
            if message.startswith(":"):
                continue
            chunk = StreamedResponseHandler._payload_of(message)
            if not chunk or chunk == "[DONE]":
                continue
            try:
                data = json.loads(chunk)
            except json.JSONDecodeError:
                # 줄이 이미 확정된 뒤이므로 이 프레임은 영구 손실이다. 조용히 넘기면
                # 토큰 손실이 그럴듯한 숫자로 위장되므로 반드시 센다.
                output.dropped_frames += 1
                continue
            if not isinstance(data, dict):
                output.dropped_frames += 1
                continue

            is_content, text, finish_reason = _extract_delta(data)
            if is_content:
                timestamp = time.perf_counter()
                if not first_chunk_received:
                    first_chunk_received = True
                    output.ttft = timestamp - st
                else:
                    output.itl.append(timestamp - most_recent_timestamp)
                most_recent_timestamp = timestamp
                generated_text += text
            if finish_reason:
                output.finish_reason = finish_reason

            completion_tokens, prompt_tokens = _extract_usage(data)
            if completion_tokens is not None:
                output.output_tokens = completion_tokens
                output.usage_reported = True
            if prompt_tokens is not None:
                # vLLM 과 동일하게 서버가 보고한 prompt_tokens 가 로컬 추정치를 덮어쓴다
                # (endpoint_request_func.py:246-247).
                output.prompt_len = prompt_tokens

    try:
        response = client.invoke_endpoint_with_response_stream(
            EndpointName=request_func_input.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        for event in response["Body"]:
            if "PayloadPart" in event:
                # NOTE: 공백만 있는 PayloadPart 도 버퍼에 넣어야 한다. 버퍼 앞에서
                # 걸러내면 SSE 구분자가 사라져 두 JSON 이 붙어버린다.
                consume(handler.add_chunk(event["PayloadPart"]["Bytes"]))
            elif "ModelStreamError" in event:
                err = event["ModelStreamError"]
                raise SageMakerStreamError(
                    f"ModelStreamError:{err.get('ErrorCode')}",
                    f"ModelStreamError: {err.get('ErrorCode')} {err.get('Message')}",
                )
            elif "InternalStreamFailure" in event:
                err = event["InternalStreamFailure"]
                raise SageMakerStreamError(
                    "InternalStreamFailure",
                    f"InternalStreamFailure: {err.get('Message')}",
                )

        consume(handler.flush())

        output.generated_text = generated_text
        # vLLM 과 동일: 마지막 콘텐츠 청크 도착 시각 - 시작 시각 (usage 전용 마지막
        # 프레임은 포함하지 않는다, endpoint_request_func.py:242/257).
        output.latency = most_recent_timestamp - st

        if output.dropped_frames:
            output.success = False
            output.error = (
                f"{output.dropped_frames} unparseable stream frame(s); "
                "token counts for this request would be wrong so it is marked failed."
            )
            output.error_kind = "StreamParseError"
        elif first_chunk_received:
            output.success = True
        else:
            output.success = False
            output.error = (
                "Never received a valid chunk to calculate TTFT."
                "This response will be marked as failed!"
            )
            output.error_kind = "NoContentChunk"

    except Exception as exc:  # noqa: BLE001 - SageMaker 오류 표면이 넓다
        # ModelError(424) / ValidationError(400) / ThrottlingException / ReadTimeoutError
        # 등은 vLLM 이 보지 못하는 오류들이다. 클래스/코드를 남겨야 429 스로틀과 모델
        # 크래시를 구분할 수 있다.
        output.success = False
        output.error_kind = _classify_error(exc)
        output.error = "".join(traceback.format_exception(*sys.exc_info()))
        # 실패한 요청도 언제까지 스트리밍했는지 남긴다(실패 비용이 보이도록).
        output.latency = max(most_recent_timestamp - st, time.perf_counter() - st)
        output.generated_text = generated_text

    return output


async def sagemaker_stream_request(
    request_func_input: RequestFuncInput,
    client: Any,
    executor: ThreadPoolExecutor,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """
    vLLM 의 `async_request_openai_completions` 대응물.

    boto3 는 blocking 이므로 전용 executor 로 offload 한다. executor 는 반드시
    --max-concurrency 크기로 명시 생성해야 한다. 기본 executor(None)는
    min(32, cpu_count+4) 로 제한되어 세마포어가 허용한 동시성이 실제로는
    스레드 수만큼밖에 나가지 않는다.
    """
    loop = asyncio.get_running_loop()
    st = time.perf_counter()
    try:
        output = await loop.run_in_executor(
            executor,
            _blocking_stream_request,
            client,
            request_func_input,
            st,
        )
    finally:
        if pbar is not None:
            # vLLM 처럼 "완료된" 요청 수를 센다(실패 포함).
            pbar.update(1)
    return output


async def wait_for_endpoint(
    request_func_input: RequestFuncInput,
    client: Any,
    executor: ThreadPoolExecutor,
    timeout_seconds: int = 600,
    retry_interval: int = 5,
) -> RequestFuncOutput:
    """
    vLLM ready_checker.wait_for_endpoint 대응물.

    SageMaker 에서는 vLLM 보다 더 중요하다: InService 라도 컨테이너가 모델을
    다 올렸다는 뜻은 아니다.
    """
    print(f"Waiting for endpoint to become up in {timeout_seconds} seconds")
    deadline = time.perf_counter() + timeout_seconds
    output = RequestFuncOutput()
    with tqdm(
        total=timeout_seconds,
        bar_format="{desc} |{bar}| {elapsed} elapsed, {remaining} remaining",
        unit="s",
    ) as pbar:
        while True:
            remaining = deadline - time.perf_counter()
            pbar.update(timeout_seconds - max(int(remaining), 0) - pbar.n)
            if remaining <= 0:
                break
            output = await sagemaker_stream_request(
                request_func_input, client, executor
            )
            if output.success:
                return output
            last_line = (output.error or "").strip().splitlines()
            print(
                "Endpoint is not ready. Error="
                f"{last_line[-1] if last_line else 'unknown'}"
            )
            await asyncio.sleep(min(retry_interval, max(remaining, 0)))
    return output


# ---------------------------------------------------------------------------
# 요청 도착 시각 생성 (vLLM serve.py:393-504 이식)
# ---------------------------------------------------------------------------


def _get_current_request_rate(
    ramp_up_strategy: Literal["linear", "exponential"] | None,
    ramp_up_start_rps: int | None,
    ramp_up_end_rps: int | None,
    request_index: int,
    total_requests: int,
    request_rate: float,
) -> float:
    if (
        ramp_up_strategy
        and ramp_up_start_rps is not None
        and ramp_up_end_rps is not None
    ):
        progress = request_index / max(total_requests - 1, 1)
        if ramp_up_strategy == "linear":
            increase = (ramp_up_end_rps - ramp_up_start_rps) * progress
            return ramp_up_start_rps + increase
        if ramp_up_strategy == "exponential":
            ratio = ramp_up_end_rps / ramp_up_start_rps
            return ramp_up_start_rps * (ratio**progress)
        raise ValueError(f"Unknown ramp-up strategy: {ramp_up_strategy}")
    return request_rate


async def get_request(
    input_requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
) -> AsyncGenerator[tuple[SampleRequest, float], None]:
    """
    지정된 rate 로 요청을 흘려보내는 제너레이터. vLLM get_request 와 동일한 수식이다.

    - 요청 간격은 Gamma(shape=burstiness, scale=1/(rate*burstiness)).
      burstiness=1 이면 Exponential(1/rate) = Poisson 과정. 고정 간격 1/rate 는
      vLLM 에서 burstiness=inf 인 특수 케이스일 뿐이다.
    - 지연을 미리 계산해 누적한 뒤, 전체 길이가 정확히 num_prompts/rate 가 되도록
      재정규화한다(시드별 throughput 흔들림 제거).
    - 절대 시각(start_ts + cumulative)까지 sleep 하므로 태스크 생성 지연이 누적되지 않는다.
    """
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    if isinstance(input_requests, Iterable) and not isinstance(input_requests, list):
        input_requests = list(input_requests)

    total_requests = len(input_requests)
    assert total_requests > 0, "No requests provided."

    # Precompute delays among requests to minimize request send laggings
    request_rates: list[float] = []
    delay_ts: list[float] = []
    for request_index in range(total_requests):
        current_request_rate = _get_current_request_rate(
            ramp_up_strategy,
            ramp_up_start_rps,
            ramp_up_end_rps,
            request_index,
            total_requests,
            request_rate,
        )
        assert current_request_rate > 0.0, (
            f"Obtained non-positive request rate {current_request_rate}."
        )
        request_rates.append(current_request_rate)
        if current_request_rate == float("inf"):
            delay_ts.append(0)
        elif burstiness == float("inf"):
            # burstiness 가 무한대면 간격은 상수 1/rate 가 된다.
            delay_ts.append(1.0 / current_request_rate)
        else:
            theta = 1.0 / (current_request_rate * burstiness)
            delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))

    for i in range(1, len(delay_ts)):
        delay_ts[i] += delay_ts[i - 1]
    if ramp_up_strategy is None and delay_ts[-1] != 0:
        # gamma 표본합은 target 대비 1-2% 오차가 생긴다. 재정규화로 그 갭을 없앤다.
        target_total_delay_s = total_requests / request_rate
        normalize_factor = target_total_delay_s / delay_ts[-1]
        delay_ts = [delay * normalize_factor for delay in delay_ts]

    # pacing 시계만 time.time() 을 쓴다(vLLM 과 동일). 측정 구간은 perf_counter.
    start_ts = time.time()
    for request_index, request in enumerate(input_requests):
        if delay_ts[request_index] > 0:
            sleep_interval_s = start_ts + delay_ts[request_index] - time.time()
            if sleep_interval_s > 0:
                await asyncio.sleep(sleep_interval_s)
        yield request, request_rates[request_index]


# ---------------------------------------------------------------------------
# 메트릭 계산 (vLLM serve.py:556-766 이식)
# ---------------------------------------------------------------------------


def calculate_metrics(
    input_requests: list[SampleRequest],
    outputs: list[RequestFuncOutput],
    dur_s: float,
    tokenizer: Any | None,
    selected_percentiles: list[float],
    goodput_config_dict: dict[str, float],
) -> tuple[BenchmarkMetrics, list[int]]:
    """vLLM calculate_metrics 와 동일한 공식. 반환값도 (metrics, actual_output_lens)."""
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: list[float] = []
    tpots: list[float] = []
    all_tpots: list[float] = []
    ttfts: list[float] = []
    e2els: list[float] = []

    truncated = 0
    eos_stopped = 0
    unknown_finish = 0
    without_usage = 0
    used_tokenizer_fallback = False
    used_constant_fallback = False

    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].output_tokens
            # falsy 검사여야 한다: usage 에 completion_tokens 가 없으면 0/None 이 되고,
            # `== 0` 비교로는 None 을 잡지 못해 집계에서 TypeError 로 런 전체가 죽는다.
            if not output_len:
                without_usage += 1
                if tokenizer is None:
                    # vLLM 과 동일한 최후 fallback. 청크 수(len(itl)+1)를 쓰지 않는 이유는
                    # 한 청크에 여러 토큰이 묶여 올 수 있어 TPOT 가 mean(ITL) 로
                    # 붕괴하기 때문이다(serve.py:589-605).
                    output_len = 1
                    used_constant_fallback = True
                else:
                    output_len = len(
                        tokenizer(
                            outputs[i].generated_text, add_special_tokens=False
                        ).input_ids
                    )
                    used_tokenizer_fallback = True
            actual_output_lens.append(output_len)
            total_input += outputs[i].prompt_len
            tpot = 0.0
            if output_len > 1:
                latency_minus_ttft = outputs[i].latency - outputs[i].ttft
                tpot = latency_minus_ttft / (output_len - 1)
                tpots.append(tpot)
            # Note: if output_len <= 1, we regard tpot as 0 for goodput
            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1

            finish_reason = outputs[i].finish_reason
            if finish_reason == "length":
                truncated += 1
            elif finish_reason in ("stop", "eos_token", "stop_sequence"):
                eos_stopped += 1
            else:
                unknown_finish += 1
        else:
            actual_output_lens.append(0)

    if goodput_config_dict:
        valid_metrics = []
        slo_values = []
        if "ttft" in goodput_config_dict:
            valid_metrics.append(ttfts)
            slo_values.append(
                goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "tpot" in goodput_config_dict:
            valid_metrics.append(all_tpots)
            slo_values.append(
                goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        if "e2el" in goodput_config_dict:
            valid_metrics.append(e2els)
            slo_values.append(
                goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            )
        for req_metric in zip(*valid_metrics):
            is_good_req = all([s >= r for s, r in zip(slo_values, req_metric)])
            if is_good_req:
                good_completed += 1

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )

    max_output_tokens_per_s = 0.0
    max_concurrent_requests = 0

    successful_outputs = [output for output in outputs if output.success]
    failed_outputs = [output for output in outputs if not output.success]

    error_kinds: dict[str, int] = {}
    for err in failed_outputs:
        kind = err.error_kind or "Unknown"
        error_kinds[kind] = error_kinds.get(kind, 0) + 1

    if failed_outputs:
        print("Failed requests during benchmark run detected (capping to 10):")
        for i, err in enumerate(failed_outputs[:10]):
            print(f"Error {i}: {err.error}")

    if successful_outputs:
        min_start_time = min(output.start_time for output in successful_outputs)
        max_end_time = max(
            output.start_time + output.latency for output in successful_outputs
        )
        duration_seconds = int(np.ceil(max_end_time - min_start_time)) + 1
        tokens_per_second = np.zeros(duration_seconds)
        concurrent_requests_per_second = np.zeros(duration_seconds)

        for output in successful_outputs:
            # 토큰 도착 시각을 start_time + ttft 에서 시작해 itl 누적으로 재구성한다.
            token_times = [output.start_time + output.ttft]
            current_time = token_times[0]
            for itl_value in output.itl:
                current_time += itl_value
                token_times.append(current_time)
            for token_time in token_times:
                second_bucket = int(token_time - min_start_time)
                if 0 <= second_bucket < duration_seconds:
                    tokens_per_second[second_bucket] += 1

            # 동시성은 설정값을 되풀이하는 게 아니라 관측된 구간에서 유도한다.
            request_start_second = int(output.start_time - min_start_time)
            request_end_second = int(
                (output.start_time + output.latency) - min_start_time
            )
            for second in range(request_start_second, request_end_second + 1):
                if 0 <= second < duration_seconds:
                    concurrent_requests_per_second[second] += 1

        if len(tokens_per_second) > 0:
            max_output_tokens_per_s = float(np.max(tokens_per_second))
            max_concurrent_requests = int(np.max(concurrent_requests_per_second))

    if used_constant_fallback:
        output_len_source = "fallback-1 (no usage frame, no tokenizer)"
    elif used_tokenizer_fallback:
        output_len_source = "tokenizer (no usage frame)"
    else:
        output_len_source = "server-usage"

    metrics = BenchmarkMetrics(
        completed=completed,
        failed=len(failed_outputs),
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        request_goodput=good_completed / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
        max_output_tokens_per_s=max_output_tokens_per_s,
        max_concurrent_requests=max_concurrent_requests,
        truncated_requests=truncated,
        eos_stopped_requests=eos_stopped,
        unknown_finish_requests=unknown_finish,
        requests_without_usage=without_usage,
        output_len_source=output_len_source,
        error_kinds=error_kinds,
    )
    return metrics, actual_output_lens


# ---------------------------------------------------------------------------
# 벤치마크 실행
# ---------------------------------------------------------------------------


async def benchmark(
    endpoint_name: str,
    region: str | None,
    endpoint_type: str,
    endpoint_url: str | None,
    model: str | None,
    input_requests: list[SampleRequest],
    tokenizer: Any | None,
    request_rate: float,
    burstiness: float,
    max_concurrency: int | None,
    disable_tqdm: bool,
    num_warmups: int,
    selected_percentile_metrics: list[str],
    selected_percentiles: list[float],
    ignore_eos: bool,
    include_usage: bool,
    goodput_config_dict: dict[str, float],
    extra_body: dict[str, Any] | None,
    request_id_prefix: str,
    ready_check_timeout_sec: int,
    read_timeout: float,
    connect_timeout: float,
    ramp_up_strategy: Literal["linear", "exponential"] | None = None,
    ramp_up_start_rps: int | None = None,
    ramp_up_end_rps: int | None = None,
) -> dict[str, Any]:
    """vLLM benchmark() 대응물. transport 만 SageMaker 로 바뀐다."""
    num_prompts = len(input_requests)

    # boto3 는 blocking 이므로 동시성 = 스레드 수. 세마포어만으로는 부족하고
    # executor 와 커넥션 풀을 같은 크기로 맞춰야 --max-concurrency 가 진짜 상한이 된다.
    effective_workers = max_concurrency if max_concurrency else num_prompts
    if effective_workers > THREAD_COUNT_WARN_THRESHOLD:
        print(
            f"WARNING: {effective_workers} worker threads will be created "
            "(boto3 has no async client, so client concurrency == thread count). "
            "GIL contention at this level adds jitter to ITL. "
            "Consider --max-concurrency."
        )

    boto_config = build_boto_config(effective_workers, read_timeout, connect_timeout)
    # 기본 세션 공유는 동시 컨텍스트에서 권장되지 않으므로 세션을 명시 생성한다.
    session = boto3.session.Session(region_name=region)
    # endpoint_url 은 로컬 검증용이다. SageMaker Runtime 프로토콜로 감싼 로컬 프록시를
    # 가리키면 vllm bench serve 와 같은 서버·같은 부하로 돌려 지표를 대조할 수 있다.
    client = session.client("sagemaker-runtime", config=boto_config,
                            endpoint_url=endpoint_url or None)
    resolved_region = session.region_name

    executor = ThreadPoolExecutor(
        max_workers=effective_workers, thread_name_prefix="sm-bench"
    )

    def make_input(request: SampleRequest, index: int) -> RequestFuncInput:
        return RequestFuncInput(
            prompt=request.prompt,
            prompt_len=request.prompt_len,
            output_len=request.expected_output_len,
            endpoint_name=endpoint_name,
            model=model,
            # request_id 는 문자열 그대로 유지한다(int 캐스팅은 비수치 id 에서 터진다).
            request_id=request.request_id or f"{request_id_prefix}{index}",
            ignore_eos=ignore_eos,
            extra_body=extra_body,
            endpoint_type=endpoint_type,
            include_usage=include_usage,
        )

    try:
        print("Starting initial single prompt test run...")
        test_input = make_input(input_requests[0], 0)

        if ready_check_timeout_sec > 0:
            test_output = await wait_for_endpoint(
                test_input,
                client,
                executor,
                timeout_seconds=ready_check_timeout_sec,
            )
            if not test_output.success:
                raise ValueError(
                    "Initial test run failed - Please make sure benchmark "
                    "arguments are correctly specified. "
                    f"Error: {test_output.error}"
                )
            print("Initial test run completed.")
        else:
            print("Skipping endpoint ready check.")

        if num_warmups > 0:
            # warmup 은 측정 구간 밖에서 실행되고 결과는 버린다. SageMaker 에서는
            # 첫 요청에 컨테이너 워밍업 + TLS 핸드셰이크 + 자격증명 조회가 얹히므로
            # vLLM 보다 오히려 더 필요하다.
            print(f"Warming up with {num_warmups} requests...")
            warmup_pbar = None if disable_tqdm else tqdm(total=num_warmups)
            warmup_semaphore = (
                asyncio.Semaphore(max_concurrency)
                if max_concurrency
                else contextlib.nullcontext()
            )

            async def warmup_limited_request_func() -> RequestFuncOutput:
                async with warmup_semaphore:
                    return await sagemaker_stream_request(
                        test_input, client, executor, warmup_pbar
                    )

            warmup_tasks = [
                asyncio.create_task(warmup_limited_request_func())
                for _ in range(num_warmups)
            ]
            _ = await asyncio.gather(*warmup_tasks)
            if warmup_pbar is not None:
                warmup_pbar.close()
            print("Warmup run completed.")

        print("Starting main benchmark run...")
        distribution = "Poisson process" if burstiness == 1.0 else "Gamma distribution"
        if ramp_up_strategy is not None:
            print(f"Traffic ramp-up strategy: {ramp_up_strategy}.")
            print(
                f"Will increase RPS from {ramp_up_start_rps} to "
                f"{ramp_up_end_rps} RPS over the duration of the benchmark."
            )
        else:
            print(f"Traffic request rate: {request_rate}")
        print(f"Burstiness factor: {burstiness} ({distribution})")
        print(f"Maximum request concurrency: {max_concurrency}")
        print(
            "SageMaker transport: endpoint="
            f"{endpoint_name} region={resolved_region} "
            f"worker_threads={effective_workers} "
            f"max_pool_connections={boto_config.max_pool_connections} "
            f"retries=off read_timeout={read_timeout}s"
        )

        pbar = None if disable_tqdm else tqdm(total=num_prompts)
        semaphore = (
            asyncio.Semaphore(max_concurrency)
            if max_concurrency
            else contextlib.nullcontext()
        )

        async def limited_request_func(
            request_func_input: RequestFuncInput,
        ) -> RequestFuncOutput:
            async with semaphore:
                return await sagemaker_stream_request(
                    request_func_input, client, executor, pbar
                )

        benchmark_start_time = time.perf_counter()
        tasks: list[asyncio.Task] = []
        index = 0
        async for request, _current_rate in get_request(
            input_requests,
            request_rate,
            burstiness,
            ramp_up_strategy,
            ramp_up_start_rps,
            ramp_up_end_rps,
        ):
            # create_task 로 즉시 디스패치해야 도착 시각이 지켜진다. 코루틴 객체만
            # 모아두면 gather 시점까지 아무 요청도 나가지 않는다.
            tasks.append(asyncio.create_task(limited_request_func(make_input(request, index))))
            index += 1

        outputs: list[RequestFuncOutput] = await asyncio.gather(*tasks)
        if pbar is not None:
            pbar.close()
        # 측정 구간은 단조 시계로만 잰다(NTP 보정에 흔들리지 않게).
        benchmark_duration = time.perf_counter() - benchmark_start_time
    finally:
        executor.shutdown(wait=False)

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    result = print_report(
        metrics=metrics,
        actual_output_lens=actual_output_lens,
        outputs=outputs,
        benchmark_duration=benchmark_duration,
        max_concurrency=max_concurrency,
        request_rate=request_rate,
        goodput_config_dict=goodput_config_dict,
        selected_percentile_metrics=selected_percentile_metrics,
        tokenizer=tokenizer,
    )
    result["region"] = resolved_region
    result["endpoint_name"] = endpoint_name
    return result


def print_report(
    metrics: BenchmarkMetrics,
    actual_output_lens: list[int],
    outputs: list[RequestFuncOutput],
    benchmark_duration: float,
    max_concurrency: int | None,
    request_rate: float,
    goodput_config_dict: dict[str, float],
    selected_percentile_metrics: list[str],
    tokenizer: Any | None,
) -> dict[str, Any]:
    """`vllm bench serve` 와 나란히 놓고 비교할 수 있도록 동일한 표를 그린다."""
    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10}".format("Failed requests:", metrics.failed))
    if max_concurrency is not None:
        print("{:<40} {:<10}".format("Maximum request concurrency:", max_concurrency))
    if request_rate != float("inf"):
        print("{:<40} {:<10.2f}".format("Request rate configured (RPS):", request_rate))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    if goodput_config_dict:
        print(
            "{:<40} {:<10.2f}".format(
                "Request goodput (req/s):", metrics.request_goodput
            )
        )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Peak output token throughput (tok/s):", metrics.max_output_tokens_per_s
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Peak concurrent requests:", metrics.max_concurrent_requests
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total token throughput (tok/s):", metrics.total_token_throughput
        )
    )

    result: dict[str, Any] = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "failed": metrics.failed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "request_goodput": metrics.request_goodput if goodput_config_dict else None,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "start_times": [output.start_time for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
        "finish_reasons": [output.finish_reason for output in outputs],
        "error_kinds": [output.error_kind for output in outputs],
        "max_output_tokens_per_s": metrics.max_output_tokens_per_s,
        "max_concurrent_requests": metrics.max_concurrent_requests,
        # SageMaker 전용 집계
        "truncated_requests": metrics.truncated_requests,
        "eos_stopped_requests": metrics.eos_stopped_requests,
        "unknown_finish_requests": metrics.unknown_finish_requests,
        "requests_without_usage": metrics.requests_without_usage,
        "output_len_source": metrics.output_len_source,
        "error_kind_counts": metrics.error_kinds,
    }

    def process_one_metric(
        metric_attribute_name: str,
        metric_name: str,
        metric_header: str,
    ) -> None:
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    # vLLM 에 없는 SageMaker 전용 섹션. 여기 값들이 없으면 "요청한 길이만큼 생성됐는가",
    # "출력 토큰 수를 신뢰할 수 있는가"를 판단할 수 없다.
    print("{s:{c}^{n}}".format(s="SageMaker Specifics", n=50, c="-"))
    print(
        "{:<40} {:<10}".format(
            "Truncated (finish_reason=length):", metrics.truncated_requests
        )
    )
    print("{:<40} {:<10}".format("Stopped at EOS:", metrics.eos_stopped_requests))
    print(
        "{:<40} {:<10}".format(
            "Finish reason unknown:", metrics.unknown_finish_requests
        )
    )
    print(
        "{:<40} {:<10}".format(
            "Requests without usage frame:", metrics.requests_without_usage
        )
    )
    print("{:<40} {:<10}".format("Output token count source:", metrics.output_len_source))
    if metrics.requests_without_usage and tokenizer is None:
        print(
            "  NOTE: the container reported no usage tokens and no --tokenizer was "
            "given, so output token counts fell back to 1 per request (vLLM's "
            "behaviour). Output/total token throughput and TPOT are NOT comparable "
            "to `vllm bench serve` under these conditions."
        )
    if metrics.error_kinds:
        print("{:<40} {:<10}".format("Error breakdown:", ""))
        for kind, count in sorted(
            metrics.error_kinds.items(), key=lambda kv: -kv[1]
        ):
            print("{:<40} {:<10}".format(f"  {kind}:", count))
    print("=" * 50)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_goodput(slo_pairs: list[str]) -> dict[str, float]:
    goodput_config_dict: dict[str, float] = {}
    try:
        for slo_pair in slo_pairs:
            slo_name, slo_val = slo_pair.split(":")
            goodput_config_dict[slo_name] = float(slo_val)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "Invalid format found for service level objectives. "
            'Specify service level objectives for goodput as "KEY:VALUE" '
            "pairs, where the key is a metric name, and the value is a "
            "number in milliseconds."
        ) from err
    return goodput_config_dict


def check_goodput_args(args: argparse.Namespace) -> dict[str, float]:
    goodput_config_dict: dict[str, float] = {}
    valid_names = ["ttft", "tpot", "e2el"]
    if args.goodput:
        goodput_config_dict = parse_goodput(args.goodput)
        for slo_name, slo_val in goodput_config_dict.items():
            if slo_name not in valid_names:
                raise ValueError(
                    f"Invalid metric name found, {slo_name}: {slo_val}. "
                    "The service level objective name should be one of "
                    f"{valid_names}. "
                )
            if slo_val < 0:
                raise ValueError(
                    f"Invalid value found, {slo_name}: {slo_val}. "
                    "The service level objective value should be non-negative."
                )
    return goodput_config_dict


def load_tokenizer(name: str | None) -> Any | None:
    """
    transformers 토크나이저 로드 (없으면 None).

    토크나이저가 없으면 (a) random 데이터셋의 입력 길이가 요청값과 달라지고,
    (b) 컨테이너가 usage 를 안 줄 때 출력 토큰 수를 셀 수 없다. 그래서 필수는
    아니지만 강력히 권장한다.
    """
    if not name:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "WARNING: --tokenizer was given but transformers is not installed; "
            "token counts will fall back to server-reported usage only."
        )
        return None
    try:
        return AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: failed to load tokenizer '{name}': {exc}")
        return None


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    # --- endpoint (SageMaker 고유) -----------------------------------------
    parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="SageMaker endpoint name (SageMaker's equivalent of vLLM's --base-url)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.environ.get("AWS_REGION"),
        help="AWS region (default: AWS_REGION env var, else the boto3 default chain)",
    )
    parser.add_argument(
        "--endpoint-url",
        type=str,
        default=os.environ.get("SM_RUNTIME_ENDPOINT_URL"),
        help="SageMaker Runtime API URL override. For local verification: point this at a "
             "proxy that speaks the SageMaker event-stream protocol over a local vLLM server, "
             "so the same load can be replayed against `vllm bench serve` and the metrics compared.",
    )
    parser.add_argument(
        "--endpoint-type",
        "--backend",
        dest="endpoint_type",
        type=str,
        default="openai",
        choices=["openai", "openai-chat", "completions", "chat"],
        help="Payload schema to send: openai (/v1/completions style, default) or "
        "openai-chat (messages). Mirrors vllm bench serve --backend.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model id sent in the payload and used for result naming. Most LMI "
        "containers ignore it; give it when the container serves several models.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer name/path. Used for token-accurate random "
        "prompts and as the output-token fallback when the container reports no "
        "usage. Defaults to --model when omitted.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label for the result file name (default: sagemaker)",
    )

    # --- transport (SageMaker 고유) ----------------------------------------
    transport_group = parser.add_argument_group("SageMaker transport")
    transport_group.add_argument(
        "--read-timeout",
        type=float,
        default=DEFAULT_READ_TIMEOUT_S,
        help=f"botocore read timeout in seconds (default: {DEFAULT_READ_TIMEOUT_S}). "
        "This is a per-socket-read timeout; the botocore default of 60s aborts "
        "requests whose first byte is slow (cold container).",
    )
    transport_group.add_argument(
        "--connect-timeout",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT_S,
        help=f"botocore connect timeout in seconds (default: "
        f"{DEFAULT_CONNECT_TIMEOUT_S})",
    )
    transport_group.add_argument(
        "--no-include-usage",
        action="store_true",
        help="Do not send stream_options.include_usage. Only for containers that "
        "reject the field; without usage, output token counts are estimates.",
    )

    # --- dataset ------------------------------------------------------------
    dataset_group = parser.add_argument_group("dataset arguments")
    dataset_group.add_argument(
        "--dataset-name",
        type=str,
        default="random",
        choices=["random", "sharegpt", "huggingface", "hf"],
        help="Dataset to use for benchmarking (default: random)",
    )
    dataset_group.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset file (sharegpt) or HuggingFace dataset ID",
    )
    dataset_group.add_argument(
        "--num-prompts",
        type=int,
        default=200,
        help="Number of prompts to process (default: 200). NOTE: vllm bench serve "
        "defaults to 1000; the smaller default here limits endpoint cost. Pass "
        "--num-prompts 1000 for exact parity.",
    )
    dataset_group.add_argument(
        "--disable-shuffle",
        action="store_true",
        help="Disable shuffling of dataset samples",
    )
    dataset_group.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)"
    )

    hf_group = parser.add_argument_group("HuggingFace dataset arguments")
    hf_group.add_argument("--hf-prompt-column", type=str, default="prompt")
    hf_group.add_argument("--hf-completion-column", type=str, default="completion")

    random_group = parser.add_argument_group("random dataset arguments")
    random_group.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request (default: 1024)",
    )
    random_group.add_argument(
        "--random-output-len",
        type=int,
        default=128,
        help="Number of output tokens per request (default: 128)",
    )
    random_group.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range ratio for sampling input/output length, used to define "
        "min/max as input_len*(1 +/- range_ratio) (default: 0.0)",
    )
    random_group.add_argument(
        "--random-prefix-len",
        type=int,
        default=0,
        help="Number of fixed prefix tokens prepended to every prompt (default: 0)",
    )

    # --- traffic / measurement ---------------------------------------------
    bench_group = parser.add_argument_group("benchmark arguments")
    bench_group.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. Also sizes the thread pool "
        "and the botocore connection pool, because boto3 is blocking. Default is "
        "None (unlimited), same as vllm bench serve.",
    )
    bench_group.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, all requests are sent "
        "at time 0. Otherwise arrival times follow a Poisson/gamma process.",
    )
    bench_group.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of the request generation (gamma shape). Only "
        "takes effect when --request-rate is not inf. 1.0 (default) is a Poisson "
        "process; <1 is burstier, >1 is more uniform.",
    )
    bench_group.add_argument(
        "--num-warmups",
        type=int,
        default=0,
        help="Number of warmup requests, excluded from all metrics (default: 0)",
    )
    bench_group.add_argument(
        "--ready-check-timeout-sec",
        type=int,
        default=0,
        help="Maximum time to wait for the endpoint to become ready, in seconds. "
        "Ready check is skipped by default. Useful on SageMaker because InService "
        "does not imply the model is loaded.",
    )
    bench_group.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Send ignore_eos so generation runs to the requested length. Forced on "
        "for --dataset-name random, matching vllm bench serve.",
    )
    bench_group.add_argument(
        "--disable-tqdm", action="store_true", help="Disable the tqdm progress bar"
    )
    bench_group.add_argument(
        "--percentile-metrics",
        type=str,
        default="ttft,tpot,itl",
        help='Comma-separated list of selected metrics to report percentiles. '
        'Allowed: "ttft", "tpot", "itl", "e2el" (default: ttft,tpot,itl)',
    )
    bench_group.add_argument(
        "--metric-percentiles",
        type=str,
        default="99",
        help='Comma-separated list of percentiles, e.g. "25,50,75" (default: "99")',
    )
    bench_group.add_argument(
        "--goodput",
        nargs="+",
        default=None,
        help='Specify SLOs for goodput as "KEY:VALUE" pairs in milliseconds, e.g. '
        '--goodput ttft:200 tpot:50. Allowed keys: ttft, tpot, e2el.',
    )
    bench_group.add_argument(
        "--ramp-up-strategy",
        type=str,
        default=None,
        choices=["linear", "exponential"],
        help="Ramp up the request rate over the run instead of holding it constant",
    )
    bench_group.add_argument("--ramp-up-start-rps", type=int, default=None)
    bench_group.add_argument("--ramp-up-end-rps", type=int, default=None)
    bench_group.add_argument(
        "--request-id-prefix",
        type=str,
        default=f"bench-{uuid.uuid4().hex[:8]}-",
        help="Prefix for generated request ids (recorded in the result JSON)",
    )

    # --- results ------------------------------------------------------------
    out_group = parser.add_argument_group("result arguments")
    out_group.add_argument(
        "--save-result", action="store_true", help="Save benchmark results to a JSON"
    )
    out_group.add_argument(
        "--save-detailed",
        action="store_true",
        help="Include per-request arrays (ttfts, itls, errors, ...) in the JSON",
    )
    out_group.add_argument(
        "--append-result",
        action="store_true",
        help="Append the result to an existing file as JSONL",
    )
    out_group.add_argument("--result-dir", type=str, default=None)
    out_group.add_argument("--result-filename", type=str, default=None)
    out_group.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        default=None,
        help="Key-value pairs recorded in the result JSON, e.g. tp=8 lmi=16",
    )

    # --- sampling parameters ------------------------------------------------
    # vLLM 과 동일하게 전부 None 기본값이고, 명시된 값만 전송한다.
    sampling_group = parser.add_argument_group("sampling parameters")
    sampling_group.add_argument("--temperature", type=float, default=None)
    sampling_group.add_argument("--top-p", type=float, default=None)
    sampling_group.add_argument("--top-k", type=int, default=None)
    sampling_group.add_argument("--min-p", type=float, default=None)
    sampling_group.add_argument("--repetition-penalty", type=float, default=None)
    sampling_group.add_argument("--presence-penalty", type=float, default=None)
    sampling_group.add_argument("--frequency-penalty", type=float, default=None)
    sampling_group.add_argument(
        "--extra-body",
        type=json.loads,
        default=None,
        help="JSON object merged into the request payload (wins over the sampling "
        "flags above)",
    )
    # 하위 호환용 deprecated 플래그. 최신 vLLM 은 샘플링 파라미터로 지원하지 않는다.
    sampling_group.add_argument(
        "--use-beam-search",
        action="store_true",
        help="DEPRECATED: forwarded as use_beam_search in the payload; modern vLLM "
        "does not accept it as a sampling parameter",
    )
    sampling_group.add_argument(
        "--best-of",
        type=int,
        default=None,
        help="DEPRECATED: forwarded as best_of in the payload",
    )


def build_extra_body(args: argparse.Namespace) -> dict[str, Any]:
    """명시된 샘플링 파라미터만 모아 payload 에 실을 dict 를 만든다."""
    sampling_params: dict[str, Any] = {}
    for name in (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    ):
        value = getattr(args, name, None)
        if value is not None:
            sampling_params[name] = value
    if args.best_of is not None:
        print("WARNING: --best-of is deprecated and may be rejected by the container.")
        sampling_params["best_of"] = args.best_of
    if args.use_beam_search:
        print(
            "WARNING: --use-beam-search is deprecated and may be rejected by the "
            "container."
        )
        sampling_params["use_beam_search"] = True

    if "temperature" not in sampling_params:
        # vLLM 이 출력하는 경고와 동일한 취지. greedy 여부가 출력 길이/EOS 시점을
        # 바꾸므로 이 차이만으로도 TPOT/throughput 이 달라진다.
        print(
            "WARNING: this tool no longer sets temperature==0 (greedy) in requests "
            "by default, matching vllm bench serve. The default will be determined "
            "on the server side and can be model/API specific. For the old "
            "behavior, include --temperature=0."
        )

    extra_body = args.extra_body or {}
    return {**sampling_params, **extra_body}


def compute_result_filename(args: argparse.Namespace, label: str, current_dt: str) -> str:
    base_model_id = (args.model or args.endpoint_name).split("/")[-1]
    max_concurrency_str = (
        f"-concurrency{args.max_concurrency}"
        if args.max_concurrency is not None
        else ""
    )
    if args.ramp_up_strategy is not None:
        file_name = (
            f"{label}-ramp-up-{args.ramp_up_strategy}-{args.ramp_up_start_rps}qps-"
            f"{args.ramp_up_end_rps}qps{max_concurrency_str}-{base_model_id}-"
            f"{current_dt}.json"
        )
    else:
        file_name = (
            f"{label}-{args.request_rate}qps{max_concurrency_str}-{base_model_id}-"
            f"{current_dt}.json"
        )
    if args.result_filename:
        file_name = args.result_filename
    if args.result_dir:
        os.makedirs(args.result_dir, exist_ok=True)
        file_name = os.path.join(args.result_dir, file_name)
    return file_name


def save_result(args: argparse.Namespace, result_json: dict[str, Any], label: str) -> None:
    current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = compute_result_filename(args, label, current_dt)
    if not args.save_detailed:
        # 데이터 포인트가 너무 많은 필드는 제거 (vLLM serve.py:2337-2351 과 동일).
        for field_name in (
            "input_lens",
            "output_lens",
            "start_times",
            "ttfts",
            "itls",
            "generated_texts",
            "errors",
            "finish_reasons",
            "error_kinds",
        ):
            result_json.pop(field_name, None)
    with open(
        file_name, mode="a+" if args.append_result else "w", encoding="utf-8"
    ) as outfile:
        if args.append_result and outfile.tell() != 0:
            outfile.write("\n")
        json.dump(result_json, outfile)
    print(f"Saved result to {file_name}")


def main() -> dict[str, Any]:
    """CLI 진입점. 측정 결과 dict 를 돌려준다.

    🔴 반환값이 있는 이유: 이 모듈은 설치해서 라이브러리로도 쓴다(pipelines/run_benchmark.py).
       None 을 돌려주면 호출자가 성공·실패를 알 수 없어, 요청이 전부 실패해도 종료 코드 0 을
       쓰게 된다 — CI 가 그것을 성공으로 읽는다(실측: 4건 전부 ValidationError 인데 exit 0).
       `completed` / `failed` 는 benchmark() 가 이미 세고 있으니 그대로 올린다.
       콘솔 스크립트(sm-bench)는 dict 를 무시하므로 CLI 동작은 달라지지 않는다.
    """
    parser = argparse.ArgumentParser(
        description="SageMaker endpoint benchmark tool with vllm bench serve parity"
    )
    add_cli_args(parser)
    args = parser.parse_args()
    print(args)

    # endpoint-type 별칭 정규화 (구 CLI 호환).
    if args.endpoint_type == "completions":
        args.endpoint_type = "openai"
    elif args.endpoint_type == "chat":
        args.endpoint_type = "openai-chat"

    if args.ramp_up_strategy is not None:
        if args.ramp_up_start_rps is None or args.ramp_up_end_rps is None:
            raise ValueError(
                "--ramp-up-start-rps and --ramp-up-end-rps must be given with "
                "--ramp-up-strategy"
            )
        if args.request_rate != float("inf"):
            raise ValueError(
                "--request-rate cannot be used with --ramp-up-strategy"
            )

    goodput_config_dict = check_goodput_args(args)

    # 시드는 데이터셋 샘플링과 도착 시각(gamma 추출) 모두에 적용해야 재현 가능하다.
    import random as _random

    _random.seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = load_tokenizer(args.tokenizer or args.model)
    if tokenizer is None:
        print(
            "NOTE: running without a tokenizer. Input lengths come from the dataset "
            "estimate (or from server-reported prompt_tokens when available), and "
            "output lengths rely on the container sending a usage frame."
        )

    ignore_eos = args.ignore_eos
    if args.dataset_name == "random" and not ignore_eos:
        # vLLM 은 random 데이터셋에서 ignore_eos 를 강제한다(serve.py:2091-2110):
        # 랜덤 프롬프트는 즉시 EOS 를 내기 때문에 요청한 출력 길이가 지켜지지 않는다.
        ignore_eos = True
        print(
            "Setting ignore_eos=True for the random dataset so generation runs to "
            "the requested length (matches vllm bench serve)."
        )

    print(f"Loading dataset: {args.dataset_name}")
    dataset = get_dataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        seed=args.seed,
        disable_shuffle=args.disable_shuffle,
        hf_prompt_column=args.hf_prompt_column,
        hf_completion_column=args.hf_completion_column,
        tokenizer=tokenizer,
    )
    input_requests = dataset.sample(
        num_requests=args.num_prompts,
        input_len=args.random_input_len,
        output_len=args.random_output_len,
        range_ratio=args.random_range_ratio,
        prefix_len=args.random_prefix_len,
        request_id_prefix=args.request_id_prefix,
    )
    print(f"Generated {len(input_requests)} requests")

    extra_body = build_extra_body(args)
    label = args.label or "sagemaker"

    benchmark_result = asyncio.run(
        benchmark(
            endpoint_name=args.endpoint_name,
            endpoint_url=args.endpoint_url,
            region=args.region,
            endpoint_type=args.endpoint_type,
            model=args.model,
            input_requests=input_requests,
            tokenizer=tokenizer,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
            num_warmups=args.num_warmups,
            selected_percentile_metrics=args.percentile_metrics.split(","),
            selected_percentiles=[
                float(p) for p in args.metric_percentiles.split(",")
            ],
            ignore_eos=ignore_eos,
            include_usage=not args.no_include_usage,
            goodput_config_dict=goodput_config_dict,
            extra_body=extra_body,
            request_id_prefix=args.request_id_prefix,
            ready_check_timeout_sec=args.ready_check_timeout_sec,
            read_timeout=args.read_timeout,
            connect_timeout=args.connect_timeout,
            ramp_up_strategy=args.ramp_up_strategy,
            ramp_up_start_rps=args.ramp_up_start_rps,
            ramp_up_end_rps=args.ramp_up_end_rps,
        )
    )

    if args.save_result or args.append_result:
        result_json: dict[str, Any] = {
            "date": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "endpoint_type": args.endpoint_type,
            "backend": args.endpoint_type,
            "label": label,
            "model_id": args.model,
            "tokenizer_id": args.tokenizer or args.model,
            "num_prompts": args.num_prompts,
        }
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=", 1)
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf"
        )
        result_json["burstiness"] = args.burstiness
        result_json["max_concurrency"] = args.max_concurrency
        result_json["ignore_eos"] = ignore_eos
        result_json["read_timeout"] = args.read_timeout
        result_json["retries"] = 0
        if args.ramp_up_strategy is not None:
            result_json["ramp_up_strategy"] = args.ramp_up_strategy
            result_json["ramp_up_start_rps"] = args.ramp_up_start_rps
            result_json["ramp_up_end_rps"] = args.ramp_up_end_rps
        result_json = {**result_json, **benchmark_result}
        save_result(args, result_json, label)

    return benchmark_result


def cli() -> None:
    """콘솔 스크립트 진입점(sm-bench). 요청이 하나도 성공하지 않으면 종료 코드 1 을 쓴다.

    표만 찍고 0 을 쓰면 호출한 스크립트가 성공으로 읽는다. main() 은 dict 를 돌려주므로
    라이브러리 사용자는 직접 판단할 수 있고, CLI 사용자를 위해 여기서 종료 코드로 바꾼다.
    """
    if not main().get("completed"):
        sys.exit(1)


if __name__ == "__main__":
    cli()
