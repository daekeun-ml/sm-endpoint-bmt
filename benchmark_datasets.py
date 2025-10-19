"""
Benchmark datasets module for SageMaker endpoint benchmarking.
Simplified version based on vLLM's datasets.py
"""
import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import numpy as np

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False


@dataclass
class SampleRequest:
    """단일 벤치마크 요청"""
    prompt: str
    prompt_len: int
    expected_output_len: int
    request_id: Optional[str] = None


class BenchmarkDataset(ABC):
    """벤치마크 데이터셋 베이스 클래스"""
    
    DEFAULT_SEED = 0
    
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        seed: int = DEFAULT_SEED,
        disable_shuffle: bool = False,
        **kwargs
    ):
        self.dataset_path = dataset_path
        self.seed = seed if seed is not None else self.DEFAULT_SEED
        self.disable_shuffle = disable_shuffle
        self.data = None
    
    @abstractmethod
    def sample(
        self,
        num_requests: int,
        request_id_prefix: str = "",
        **kwargs
    ) -> List[SampleRequest]:
        """샘플 요청 생성"""
        raise NotImplementedError


class RandomDataset(BenchmarkDataset):
    """
    랜덤 합성 데이터셋
    vLLM의 RandomDataset을 단순화한 버전
    """
    
    DEFAULT_INPUT_LEN = 1024
    DEFAULT_OUTPUT_LEN = 128
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rng = np.random.default_rng(self.seed)
    
    def sample(
        self,
        num_requests: int,
        request_id_prefix: str = "",
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        **kwargs
    ) -> List[SampleRequest]:
        """랜덤 프롬프트 생성"""
        requests = []
        
        # 간단한 단어 리스트로 프롬프트 생성
        words = [
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "up", "about", "into", "through", "during", "before",
            "after", "above", "below", "between", "under", "again", "further",
            "then", "once", "here", "there", "when", "where", "why", "how",
            "all", "both", "each", "few", "more", "most", "other", "some",
            "such", "no", "nor", "not", "only", "own", "same", "so", "than"
        ]
        
        for i in range(num_requests):
            # 대략적인 토큰 수로 단어 수 계산 (평균 1.3 토큰/단어)
            num_words = int(input_len / 1.3)
            prompt = " ".join(self._rng.choice(words, size=num_words))
            
            requests.append(
                SampleRequest(
                    prompt=prompt,
                    prompt_len=input_len,
                    expected_output_len=output_len,
                    request_id=request_id_prefix + str(i)
                )
            )
        
        return requests


class ShareGPTDataset(BenchmarkDataset):
    """ShareGPT 형식 데이터셋"""
    
    def load_data(self):
        """JSON 파일에서 데이터 로드"""
        if self.dataset_path is None:
            raise ValueError("dataset_path must be provided for ShareGPT dataset")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def sample(
        self,
        num_requests: int,
        request_id_prefix: str = "",
        **kwargs
    ) -> List[SampleRequest]:
        """ShareGPT 데이터에서 샘플링"""
        if self.data is None:
            self.load_data()
        
        # 랜덤 시드 설정
        random.seed(self.seed)
        
        # 데이터 셔플 또는 순차 선택
        if self.disable_shuffle:
            sampled_data = self.data[:num_requests]
        else:
            sampled_data = random.sample(
                self.data,
                min(num_requests, len(self.data))
            )
        
        requests = []
        for i, conversation in enumerate(sampled_data):
            # ShareGPT 형식: conversations 리스트에서 첫 user 메시지 추출
            if 'conversations' in conversation:
                user_messages = [
                    msg['value'] for msg in conversation['conversations']
                    if msg['from'] == 'human' or msg['from'] == 'user'
                ]
                assistant_messages = [
                    msg['value'] for msg in conversation['conversations']
                    if msg['from'] == 'gpt' or msg['from'] == 'assistant'
                ]
                
                if user_messages and assistant_messages:
                    prompt = user_messages[0]
                    # 대략적인 토큰 수 계산 (단어 수 * 1.3)
                    prompt_len = int(len(prompt.split()) * 1.3)
                    output_len = int(len(assistant_messages[0].split()) * 1.3)
                    
                    requests.append(
                        SampleRequest(
                            prompt=prompt,
                            prompt_len=prompt_len,
                            expected_output_len=output_len,
                            request_id=request_id_prefix + str(i)
                        )
                    )
        
        # 부족하면 오버샘플링
        if len(requests) < num_requests:
            additional_needed = num_requests - len(requests)
            for i in range(additional_needed):
                req = random.choice(requests)
                requests.append(
                    SampleRequest(
                        prompt=req.prompt,
                        prompt_len=req.prompt_len,
                        expected_output_len=req.expected_output_len,
                        request_id=request_id_prefix + str(len(requests))
                    )
                )
        
        return requests[:num_requests]


class HuggingFaceDataset(BenchmarkDataset):
    """HuggingFace 데이터셋"""
    
    def __init__(
        self,
        dataset_path: str,
        prompt_column: str = "prompt",
        completion_column: str = "completion",
        **kwargs
    ):
        super().__init__(dataset_path=dataset_path, **kwargs)
        self.prompt_column = prompt_column
        self.completion_column = completion_column
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets library is required. "
                "Install it with: pip install datasets"
            )
    
    def load_data(self):
        """HuggingFace에서 데이터셋 로드"""
        if self.dataset_path is None:
            raise ValueError("dataset_path (HuggingFace dataset ID) must be provided")
        
        print(f"Loading HuggingFace dataset: {self.dataset_path}")
        
        # HuggingFace dataset 로드
        # dataset_path는 "org/dataset-name" 형식
        dataset = load_dataset(self.dataset_path, split="train")
        self.data = list(dataset)
        
        print(f"Loaded {len(self.data)} samples from HuggingFace")
    
    def sample(
        self,
        num_requests: int,
        request_id_prefix: str = "",
        **kwargs
    ) -> List[SampleRequest]:
        """HuggingFace 데이터에서 샘플링"""
        if self.data is None:
            self.load_data()
        
        # 랜덤 시드 설정
        random.seed(self.seed)
        
        # 데이터 셔플 또는 순차 선택
        if self.disable_shuffle:
            sampled_data = self.data[:num_requests]
        else:
            sampled_data = random.sample(
                self.data,
                min(num_requests, len(self.data))
            )
        
        requests = []
        for i, item in enumerate(sampled_data):
            # prompt와 completion 컬럼 추출
            prompt = item.get(self.prompt_column, "")
            completion = item.get(self.completion_column, "")
            
            if not prompt:
                # 다른 일반적인 컬럼명 시도
                prompt = item.get("text", item.get("input", item.get("question", "")))
            
            if not completion:
                # 다른 일반적인 컬럼명 시도
                completion = item.get("output", item.get("answer", item.get("response", "")))
            
            if prompt:
                # 대략적인 토큰 수 계산 (단어 수 * 1.3)
                prompt_len = int(len(str(prompt).split()) * 1.3)
                output_len = int(len(str(completion).split()) * 1.3) if completion else 128
                
                requests.append(
                    SampleRequest(
                        prompt=str(prompt),
                        prompt_len=prompt_len,
                        expected_output_len=output_len,
                        request_id=request_id_prefix + str(i)
                    )
                )
        
        # 부족하면 오버샘플링
        if len(requests) < num_requests:
            additional_needed = num_requests - len(requests)
            for i in range(additional_needed):
                req = random.choice(requests)
                requests.append(
                    SampleRequest(
                        prompt=req.prompt,
                        prompt_len=req.prompt_len,
                        expected_output_len=req.expected_output_len,
                        request_id=request_id_prefix + str(len(requests))
                    )
                )
        
        return requests[:num_requests]


def get_dataset(
    dataset_name: str,
    dataset_path: Optional[str] = None,
    seed: int = 0,
    disable_shuffle: bool = False,
    hf_prompt_column: str = "prompt",
    hf_completion_column: str = "completion",
) -> BenchmarkDataset:
    """데이터셋 팩토리 함수"""
    dataset_name = dataset_name.lower()
    
    if dataset_name == "random":
        return RandomDataset(
            seed=seed,
            disable_shuffle=disable_shuffle
        )
    elif dataset_name == "sharegpt":
        return ShareGPTDataset(
            dataset_path=dataset_path,
            seed=seed,
            disable_shuffle=disable_shuffle
        )
    elif dataset_name == "huggingface" or dataset_name == "hf":
        return HuggingFaceDataset(
            dataset_path=dataset_path,
            seed=seed,
            disable_shuffle=disable_shuffle,
            prompt_column=hf_prompt_column,
            completion_column=hf_completion_column
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
