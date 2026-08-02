"""
Benchmark datasets module for SageMaker endpoint benchmarking.

vLLM `vllm/benchmarks/datasets/` 의 축소판이다. 클래스 수는 훨씬 적지만
`SampleRequest` 시그니처와 길이 샘플링 수식은 vLLM 과 맞춰서, 나중에 데이터셋만
바꿔 끼워도 SageMaker transport 를 건드릴 필요가 없게 한다.

토크나이저가 주어지면 RandomDataset 은 vLLM 과 동일하게 실제 토큰 id 를 만들고
decode→encode 왕복으로 목표 토큰 수를 맞춘다. 토크나이저가 없으면 단어 기반
근사로 떨어지는데, 이 경우 프롬프트 길이는 "요청값"이 아니라 "추정값"이라고
표시한다(조용히 틀린 숫자를 보고하지 않기 위해).
"""

from __future__ import annotations

import json
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from datasets import load_dataset

    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


# 토크나이저 없이 동작할 때 쓰는 단어→토큰 환산 계수. 실제 비율은 모델마다 다르므로
# 이 경로에서는 prompt_len 이 어디까지나 추정치임을 호출자에게 알린다.
WORDS_TO_TOKENS = 1.3


@dataclass
class SampleRequest:
    """단일 벤치마크 요청. vLLM SampleRequest 의 부분집합."""

    prompt: str
    prompt_len: int
    expected_output_len: int
    request_id: str | None = None
    # prompt_len 이 실제 토큰 수인지(True) 단어 기반 추정치인지(False).
    prompt_len_is_exact: bool = False


class BenchmarkDataset(ABC):
    """벤치마크 데이터셋 베이스 클래스."""

    DEFAULT_SEED = 0

    def __init__(
        self,
        dataset_path: str | None = None,
        seed: int = DEFAULT_SEED,
        disable_shuffle: bool = False,
        tokenizer: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.dataset_path = dataset_path
        self.seed = seed if seed is not None else self.DEFAULT_SEED
        self.disable_shuffle = disable_shuffle
        self.tokenizer = tokenizer
        self.data: list[Any] | None = None
        # 전역 random 모듈을 건드리지 않는다. sample() 안에서 random.seed() 를 호출하면
        # 무관한 난수(도착 시각 등)까지 흔들린다.
        self._py_rng = random.Random(self.seed)

    @abstractmethod
    def sample(
        self,
        num_requests: int,
        request_id_prefix: str = "",
        **kwargs: Any,
    ) -> list[SampleRequest]:
        """샘플 요청 생성."""
        raise NotImplementedError

    # --- 공통 유틸 ---------------------------------------------------------

    def _count_tokens(self, text: str) -> tuple[int, bool]:
        """(토큰 수, 정확한지) 를 반환."""
        if self.tokenizer is not None:
            return (
                len(self.tokenizer(text, add_special_tokens=False).input_ids),
                True,
            )
        return int(len(text.split()) * WORDS_TO_TOKENS), False

    def _oversample(
        self,
        requests: list[SampleRequest],
        num_requests: int,
        request_id_prefix: str,
        dataset_hint: str,
    ) -> list[SampleRequest]:
        """모자란 만큼 복제해 채운다. 빈 리스트는 명확한 오류로 알린다."""
        if not requests:
            raise ValueError(
                f"{dataset_hint} produced no usable requests. Check the dataset "
                "path and the expected columns."
            )
        while len(requests) < num_requests:
            req = self._py_rng.choice(requests)
            requests.append(
                SampleRequest(
                    prompt=req.prompt,
                    prompt_len=req.prompt_len,
                    expected_output_len=req.expected_output_len,
                    request_id=request_id_prefix + str(len(requests)),
                    prompt_len_is_exact=req.prompt_len_is_exact,
                )
            )
        return requests[:num_requests]


class RandomDataset(BenchmarkDataset):
    """
    합성 데이터셋.

    토크나이저가 있으면 vLLM RandomDataset 과 동일한 절차를 따른다:
      1) 요청별 입력/출력 길이를 정수 균일분포에서 추출 (range_ratio).
      2) 고정 prefix (prefix_len) 를 앞에 붙인다.
      3) 본문 토큰을 (offset + index + arange(input_len)) % vocab_size 로 만들고
         special token 은 제외한다.
      4) decode → encode 왕복으로 실제 토큰 수를 목표에 맞춘다.
    """

    DEFAULT_PREFIX_LEN = 0
    DEFAULT_RANGE_RATIO = 0.0
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128

    # 토크나이저 없이 쓰는 fallback 단어 사전. 어휘가 작으면 prefix cache hit 이
    # 부풀려지므로 최소한의 다양성만 확보한다.
    _FALLBACK_WORDS = [
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
        "by", "from", "up", "about", "into", "through", "during", "before",
        "after", "above", "below", "between", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # 전역 상태와 격리된 RNG (vLLM 과 동일한 규칙).
        self._rng = np.random.default_rng(self.seed)

    def sample(
        self,
        num_requests: int,
        request_id_prefix: str = "",
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        range_ratio: float = DEFAULT_RANGE_RATIO,
        prefix_len: int = DEFAULT_PREFIX_LEN,
        **kwargs: Any,
    ) -> list[SampleRequest]:
        if not 0.0 <= range_ratio < 1.0:
            raise ValueError("range_ratio must be in [0, 1).")

        if self.tokenizer is None:
            return self._sample_words(
                num_requests, request_id_prefix, input_len, output_len, range_ratio
            )
        return self._sample_tokens(
            num_requests,
            request_id_prefix,
            input_len,
            output_len,
            range_ratio,
            prefix_len,
        )

    # --- 토크나이저 기반 (정확) -------------------------------------------

    def _sample_tokens(
        self,
        num_requests: int,
        request_id_prefix: str,
        input_len: int,
        output_len: int,
        range_ratio: float,
        prefix_len: int,
    ) -> list[SampleRequest]:
        tokenizer = self.tokenizer
        num_special = int(tokenizer.num_special_tokens_to_add())
        real_input_len = max(0, int(input_len) - num_special)

        input_low = math.floor(real_input_len * (1 - range_ratio))
        input_high = math.ceil(real_input_len * (1 + range_ratio))
        output_low = max(math.floor(output_len * (1 - range_ratio)), 1)
        output_high = max(math.ceil(output_len * (1 + range_ratio)), 1)
        if prefix_len + input_low < 1:
            raise ValueError(
                "--random-input-len is too small: with "
                f"{num_special} special tokens and range ratio {range_ratio}, the "
                f"minimum total input length is {prefix_len + input_low}."
            )
        print(
            f"Sampling input_len from [{input_low}, {input_high}] and output_len "
            f"from [{output_low}, {output_high}]"
        )

        input_lens = self._rng.integers(input_low, input_high + 1, size=num_requests)
        output_lens = self._rng.integers(
            output_low, output_high + 1, size=num_requests
        )
        offsets = self._rng.integers(0, tokenizer.vocab_size, size=num_requests)

        vocab_size = tokenizer.vocab_size
        prohibited = set(tokenizer.all_special_ids or [])
        allowed_tokens = np.array(
            [t for t in range(vocab_size) if t not in prohibited]
        )

        prefix_token_ids = (
            self._rng.choice(allowed_tokens, size=prefix_len).tolist()
            if prefix_len > 0
            else []
        )

        requests: list[SampleRequest] = []
        token_mismatch_total = 0
        for i in range(num_requests):
            body_len = int(input_lens[i])
            inner = (int(offsets[i]) + i + np.arange(body_len)) % len(allowed_tokens)
            token_sequence = prefix_token_ids + allowed_tokens[inner].tolist()
            prompt, token_sequence, mismatch = self._decode_to_target_len(
                token_sequence, prefix_len + body_len
            )
            token_mismatch_total += mismatch
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=len(token_sequence),
                    expected_output_len=int(output_lens[i]),
                    request_id=request_id_prefix + str(i),
                    prompt_len_is_exact=True,
                )
            )

        if token_mismatch_total != 0:
            sign = "more" if token_mismatch_total > 0 else "fewer"
            print(
                f"WARNING: across all generated prompts there were "
                f"{abs(token_mismatch_total)} {sign} tokens than expected after "
                "decoding and re-encoding. This is expected due to the imperfect "
                "nature of the sampling procedure."
            )
        return requests

    def _decode_to_target_len(
        self,
        token_sequence: list[int],
        target_token_len: int,
        max_retry: int = 10,
    ) -> tuple[str, list[int], int]:
        """
        decode → encode 왕복으로 실제 토큰 수를 목표에 맞춘다.

        토크나이저가 연속 토큰과 decode-then-encode 결과의 1:1 대응을 보장하지 않기
        때문에 필요하다(vLLM gen_prompt_decode_to_target_len 와 동일한 로직).
        """
        tokenizer = self.tokenizer
        remain = max_retry
        mismatch = 0
        while True:
            prompt = tokenizer.decode(token_sequence)
            token_sequence = tokenizer.encode(prompt, add_special_tokens=False)
            if remain <= 0:
                if len(token_sequence) != target_token_len:
                    mismatch = len(token_sequence) - target_token_len
                break
            if len(token_sequence) == target_token_len:
                break
            if len(token_sequence) < target_token_len:
                token_sequence.extend(
                    self._rng.integers(
                        0,
                        tokenizer.vocab_size,
                        size=target_token_len - len(token_sequence),
                    ).tolist()
                )
            else:
                token_sequence = token_sequence[:target_token_len]
            remain -= 1
        return prompt, token_sequence, mismatch

    # --- 토크나이저 없음 (근사) -------------------------------------------

    def _sample_words(
        self,
        num_requests: int,
        request_id_prefix: str,
        input_len: int,
        output_len: int,
        range_ratio: float,
    ) -> list[SampleRequest]:
        print(
            "WARNING: no tokenizer available, so random prompts are built from a "
            "word list and their token length is only an estimate "
            f"(~{WORDS_TO_TOKENS} tokens/word). Pass --tokenizer for token-accurate "
            "input lengths comparable to `vllm bench serve`."
        )
        input_low = math.floor(input_len * (1 - range_ratio))
        input_high = math.ceil(input_len * (1 + range_ratio))
        output_low = max(math.floor(output_len * (1 - range_ratio)), 1)
        output_high = max(math.ceil(output_len * (1 + range_ratio)), 1)
        input_lens = self._rng.integers(input_low, input_high + 1, size=num_requests)
        output_lens = self._rng.integers(
            output_low, output_high + 1, size=num_requests
        )

        requests: list[SampleRequest] = []
        for i in range(num_requests):
            num_words = max(int(int(input_lens[i]) / WORDS_TO_TOKENS), 1)
            prompt = " ".join(self._rng.choice(self._FALLBACK_WORDS, size=num_words))
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    # 실제로 얼마나 되는지 알 수 없으므로 추정치임을 표시한다.
                    prompt_len=int(num_words * WORDS_TO_TOKENS),
                    expected_output_len=int(output_lens[i]),
                    request_id=request_id_prefix + str(i),
                    prompt_len_is_exact=False,
                )
            )
        return requests


class ShareGPTDataset(BenchmarkDataset):
    """ShareGPT 형식 데이터셋."""

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for ShareGPT dataset")
        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)

    def sample(
        self,
        num_requests: int,
        request_id_prefix: str = "",
        output_len: int | None = None,
        **kwargs: Any,
    ) -> list[SampleRequest]:
        if self.data is None:
            self.load_data()

        if self.disable_shuffle:
            sampled_data = self.data[:num_requests]
        else:
            sampled_data = self._py_rng.sample(
                self.data, min(num_requests, len(self.data))
            )

        requests: list[SampleRequest] = []
        for i, conversation in enumerate(sampled_data):
            if "conversations" not in conversation:
                continue
            user_messages = [
                msg["value"]
                for msg in conversation["conversations"]
                if msg["from"] in ("human", "user")
            ]
            assistant_messages = [
                msg["value"]
                for msg in conversation["conversations"]
                if msg["from"] in ("gpt", "assistant")
            ]
            if not (user_messages and assistant_messages):
                continue
            prompt = user_messages[0]
            prompt_len, exact = self._count_tokens(prompt)
            completion_len, _ = self._count_tokens(assistant_messages[0])
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=prompt_len,
                    expected_output_len=output_len or completion_len,
                    request_id=request_id_prefix + str(i),
                    prompt_len_is_exact=exact,
                )
            )

        return self._oversample(
            requests, num_requests, request_id_prefix, "ShareGPT dataset"
        )


class HuggingFaceDataset(BenchmarkDataset):
    """일반 prompt/completion 컬럼을 읽는 HuggingFace 데이터셋 리더."""

    DEFAULT_OUTPUT_LEN = 128

    def __init__(
        self,
        dataset_path: str,
        prompt_column: str = "prompt",
        completion_column: str = "completion",
        **kwargs: Any,
    ) -> None:
        super().__init__(dataset_path=dataset_path, **kwargs)
        self.prompt_column = prompt_column
        self.completion_column = completion_column
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets library is required. "
                "Install it with: pip install datasets"
            )

    def load_data(self) -> None:
        if self.dataset_path is None:
            raise ValueError("dataset_path (HuggingFace dataset ID) must be provided")
        print(f"Loading HuggingFace dataset: {self.dataset_path}")
        dataset = load_dataset(self.dataset_path, split="train")
        self.data = list(dataset)
        print(f"Loaded {len(self.data)} samples from HuggingFace")

    def sample(
        self,
        num_requests: int,
        request_id_prefix: str = "",
        output_len: int | None = None,
        **kwargs: Any,
    ) -> list[SampleRequest]:
        if self.data is None:
            self.load_data()

        if self.disable_shuffle:
            sampled_data = self.data[:num_requests]
        else:
            sampled_data = self._py_rng.sample(
                self.data, min(num_requests, len(self.data))
            )

        requests: list[SampleRequest] = []
        for i, item in enumerate(sampled_data):
            prompt = item.get(self.prompt_column) or item.get(
                "text", item.get("input", item.get("question", ""))
            )
            completion = item.get(self.completion_column) or item.get(
                "output", item.get("answer", item.get("response", ""))
            )
            if not prompt:
                continue
            prompt_len, exact = self._count_tokens(str(prompt))
            if output_len is not None:
                expected = output_len
            elif completion:
                expected, _ = self._count_tokens(str(completion))
            else:
                expected = self.DEFAULT_OUTPUT_LEN
            requests.append(
                SampleRequest(
                    prompt=str(prompt),
                    prompt_len=prompt_len,
                    expected_output_len=max(expected, 1),
                    request_id=request_id_prefix + str(i),
                    prompt_len_is_exact=exact,
                )
            )

        return self._oversample(
            requests,
            num_requests,
            request_id_prefix,
            f"HuggingFace dataset '{self.dataset_path}' "
            f"(columns '{self.prompt_column}'/'{self.completion_column}')",
        )


def get_dataset(
    dataset_name: str,
    dataset_path: str | None = None,
    seed: int = 0,
    disable_shuffle: bool = False,
    hf_prompt_column: str = "prompt",
    hf_completion_column: str = "completion",
    tokenizer: Any | None = None,
) -> BenchmarkDataset:
    """데이터셋 팩토리 함수."""
    dataset_name = dataset_name.lower()

    if dataset_name == "random":
        return RandomDataset(
            seed=seed, disable_shuffle=disable_shuffle, tokenizer=tokenizer
        )
    if dataset_name == "sharegpt":
        return ShareGPTDataset(
            dataset_path=dataset_path,
            seed=seed,
            disable_shuffle=disable_shuffle,
            tokenizer=tokenizer,
        )
    if dataset_name in ("huggingface", "hf"):
        return HuggingFaceDataset(
            dataset_path=dataset_path,
            seed=seed,
            disable_shuffle=disable_shuffle,
            tokenizer=tokenizer,
            prompt_column=hf_prompt_column,
            completion_column=hf_completion_column,
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")
