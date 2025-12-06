from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
import os
import random
import time
from typing import List, Iterable, Optional, Any, Callable, Dict

from openai import OpenAI
from openai import RateLimitError

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError:
    raise ImportError("vLLM is not installed. Please install it with 'pip install vllm'")
import threading
_vllm_init_lock = threading.Lock()

def retry_on_ratelimit(max_retries=5, base_delay=1.0, max_delay=10.0):
    def deco(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except RateLimitError:
                    if attempt == max_retries - 1:
                        raise
                    sleep = min(max_delay, delay) + random.uniform(0, delay)
                    time.sleep(sleep)
                    delay *= 2
        return wrapped
    return deco


class LLMClient(ABC):
    """
    Abstract base class for calling LLM APIs.
    Provides a default call_batch implementation that calls call_api in parallel.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    @abstractmethod
    def call_api(self, prompt: str, **kwargs) -> str:
        """
        Call the underlying LLM API with a single prompt.
        Must be implemented by subclasses.

        Args:
            prompt: prompt string
            kwargs: vendor-specific options

        Returns:
            response string
        """
        raise NotImplementedError

    def call_batch(
        self,
        prompts: Iterable[str],
        max_workers: int = 8,
        chunk_size: Optional[int] = None,
        raise_on_error: bool = False,
        per_result_callback: Optional[Callable[[int, Optional[str], Optional[Exception]], Any]] = None,
        **call_api_kwargs,
    ) -> List[Optional[str]]:
        """
        Default batch implementation: runs call_api in parallel with a ThreadPoolExecutor.
        Preserves order of input prompts in returned list.

        Args:
            prompts: iterable of prompt strings
            max_workers: max parallel workers
            chunk_size: if set, splits the prompts into chunks of this size and processes sequentially.
                        Useful to limit concurrency or rate.
            raise_on_error: if True, re-raise the exception when any prompt fails after retries.
            per_result_callback: optional function called as callback(idx, result, exception) for each finished prompt.
            call_api_kwargs: forwarded to call_api.

        Returns:
            list of responses (or None for failed items if raise_on_error is False)
        """
        prompts = list(prompts)
        results: List[Optional[str]] = [None] * len(prompts)

        def _submit_range(start_idx: int, end_idx: int):
            # submit jobs for a slice [start_idx, end_idx)
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for i in range(start_idx, end_idx):
                    fut = ex.submit(self.call_api, prompts[i], **call_api_kwargs)
                    futures[fut] = i
                for fut in as_completed(futures):
                    idx = futures[fut]
                    try:
                        res = fut.result()
                        results[idx] = res
                        if per_result_callback:
                            per_result_callback(idx, res, None)
                    except Exception as exc:
                        # If caller wants to raise, do so; otherwise set None and continue
                        if per_result_callback:
                            per_result_callback(idx, None, exc)
                        if raise_on_error:
                            raise
                        results[idx] = None

        if chunk_size is None or chunk_size <= 0:
            # one-shot submit all prompts (bounded by max_workers in each executor)
            _submit_range(0, len(prompts))
        else:
            # process chunks sequentially to throttle
            for start in range(0, len(prompts), chunk_size):
                end = min(start + chunk_size, len(prompts))
                _submit_range(start, end)

        return results


class NvidiaLLMClient(LLMClient):
    """
    Concrete implementation of LLMClient for the NVIDIA API (non-streaming).
    """

    def __init__(self, config: dict):
        super().__init__(config)
        api_key = config.get("api_key") or os.environ.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError(
                "API key for NVIDIA must be provided in config['api_key'] or NVIDIA_API_KEY env var."
            )

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model_name = config.get("model_name", "google/gemma-3-1b-it")

        gen_conf = config.get("generation_config", {})
        self.temperature = gen_conf.get("temperature", 0)
        self.top_p = gen_conf.get("top_p", 0.7)
        self.max_tokens = gen_conf.get("max_tokens", 8192)

    def set_model(self, model_name: str):
        self.model_name = model_name

    @retry_on_ratelimit(max_retries=20, base_delay=0.5, max_delay=5.0)
    def call_api(self, prompt: str, **kwargs) -> str:
        """
        Single prompt call (non-streaming).
        kwargs are forwarded to the underlying API call if needed.
        """
        # print("prompt:", prompt)  # keep optional for debugging
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role":"system","content":"Reasoning: high"},{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"thinking": True}},
            # any additional kwargs can be merged here if needed
        )
        return response.choices[0].message.content

    # Optionally override call_batch if the vendor supports true batched calls.
    # For now, we inherit the default implementation from LLMClient.

class VLLMClient(LLMClient):
    def __init__(self, config: dict):
        super().__init__(config)
        
        model_name = config.get("model_name")
        if not model_name:
            raise ValueError("model_name must be provided in the config for VLLMClient")

        # vLLM-specific engine arguments (e.g., for multi-GPU)
        engine_args = config.get("engine_args", {})

        # Check if pre-defined loras exist in config to auto-enable LoRA support
        self.lora_config = config.get("lora_modules", {}) # Format: {"name": "path/to/lora"}
        print("LoRA modules to load:", self.lora_config)
        if self.lora_config or config.get("enable_lora", False):
            engine_args["enable_lora"] = True
            # Optional: Tune these based on VRAM (defaults usually work)
            # engine_args["max_loras"] = 4 
            # engine_args["max_lora_rank"] = 64
        print(f"Initializing vLLM model '{model_name}' with engine args: {engine_args}")
        # Guard initialization with a global lock to avoid races
        with _vllm_init_lock:
            self.llm = LLM(model=model_name, **engine_args)

        # 2. Initialize LoRA Registry
        # vLLM requires a unique integer ID for every loaded adapter.
        self.lora_requests: Dict[str, LoRARequest] = {}
        self._lora_id_counter = 1
        
        # Load adapters defined in config
        for name, path in self.lora_config.items():
            self.load_adapter(name, path)

        # per-instance lock to serialize calls to self.llm.generate(...)
        self._generate_lock = threading.Lock()
        
        # Default generation config
        gen_conf = config.get("generation_config", {})
        self.temperature = gen_conf.get("temperature", 0.0)
        self.top_p = gen_conf.get("top_p", 1.0)
        self.max_tokens = gen_conf.get("max_tokens", 512)
        self.stop_sequences = gen_conf.get("stop", [])
        self.enable_thinking = config.get("enable_thinking", False)

    def load_adapter(self, name: str, path: str):
        """
        Registers a LoRA adapter so it can be called by name.
        """
        if name in self.lora_requests:
            return # Already loaded
        
        # Create the vLLM request object with a unique ID
        self.lora_requests[name] = LoRARequest(
            lora_name=name,
            lora_int_id=self._lora_id_counter,
            lora_path=path
        )
        self._lora_id_counter += 1

    def _get_lora_request(self, adapter_name: Optional[str]) -> Optional[LoRARequest]:
        """Helper to retrieve the LoRA object or None."""
        if not adapter_name:
            return None
        
        if adapter_name not in self.lora_requests:
            print(f"LoRA adapter '{adapter_name}' not found. Available: {list(self.lora_requests.keys())}")
            return None
        
        return self.lora_requests[adapter_name]

    def _format_prompt_with_thinking(self, prompt: str) -> str:
        if self.enable_thinking:
            return f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            return f"{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    
    def _create_sampling_params(self, **kwargs) -> SamplingParams:
        return SamplingParams(
            seed        = kwargs.get("seed", self.config.get("seed", None)),
            temperature = kwargs.get("temperature", self.temperature),
            top_p       = kwargs.get("top_p", self.top_p),
            max_tokens  = kwargs.get("max_tokens", self.max_tokens),
            stop        = kwargs.get("stop", self.stop_sequences),
        )
    
    def call_api(self, prompt: str, adapter_name: str = None, **kwargs) -> str:
        """
        :param adapter_name: The string name of the LoRA to use (must be loaded first).
        """
        sampling_params = self._create_sampling_params(**kwargs)
        prompt = self._format_prompt_with_thinking(prompt)
        
        # Retrieve the specific LoRA object
        lora_req = self._get_lora_request(adapter_name)

        with self._generate_lock:
            # Pass lora_request to generate
            outputs = self.llm.generate(
                [prompt], 
                sampling_params, 
                lora_request=lora_req
            )
        return outputs[0].outputs[0].text

    def call_batch(
        self,
        prompts: Iterable[str],
        adapter_name: str = None,
        **call_api_kwargs,
    ) -> List[Optional[str]]:
        """
        Note: vLLM applies the SAME LoRA to the entire batch in this implementation.
        """
        prompts = list(prompts)
        sampling_params = self._create_sampling_params(**call_api_kwargs)
        prompts = [self._format_prompt_with_thinking(p) for p in prompts]
        
        # Retrieve the specific LoRA object
        lora_req = self._get_lora_request(adapter_name)
        print(f"Using LoRA adapter: {adapter_name}")
        with self._generate_lock:
            outputs = self.llm.generate(
                prompts, 
                sampling_params, 
                lora_request=lora_req
            )

        results = [output.outputs[0].text for output in outputs]
        
        return results